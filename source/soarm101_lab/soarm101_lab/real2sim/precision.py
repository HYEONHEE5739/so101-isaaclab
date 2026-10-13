"""Evidence-based raw-register joint coordinates. No engine or hardware imports.

Present_Position(normalize=False) already includes the motor's Homing_Offset.
Never subtract that offset again. Profiles are software transforms, not motor files.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
import xml.etree.ElementTree as ET
import numpy as np
from soarm101_lab.so101_joint_mapping import JOINTS, LEGACY_MAPPER
from .storage import atomic

SCHEMA = 'so101.precision-calibration/1'
MAPPING = 'so101.precision-joint-mapping/1'
PROFILE_ROOT = Path(__file__).resolve().parents[4] / 'outputs/real2sim/joint_calibration'


def active_precision(role, root=None):
    """Read the approved mapping; no device access or automatic activation."""
    return ProfileStore(root or PROFILE_ROOT).active(role)


def collection_coordinates(contract, mapper):
    """Keep model radians separate from the hardware transform used to collect them."""
    if mapper is None: return copy.deepcopy(contract)
    return {**copy.deepcopy(contract), 'hardware_mapping': mapper.metadata(), 'simulation_joint_frame': 'urdf'}


def uses_urdf_frame(contract):
    frame = (contract or {}).get('simulation_joint_frame')
    if frame is None: return False  # Historical data keep their historical scene.
    if frame != 'urdf': raise ValueError('Unknown recorded simulation joint frame')
    mapper_from_metadata(contract['hardware_mapping'])
    if contract['hardware_mapping'].get('schema') != MAPPING:
        raise ValueError('URDF calibration frame requires precision collection evidence')
    return True


def require_hardware_policy_frame(contract, profile, mapper):
    if mapper is None or uses_urdf_frame(contract): return
    from ..so101_robot_spec import simulation_joint_offsets
    if np.any(abs(simulation_joint_offsets(profile.get('simulation_joint_offsets_rad'), JOINTS)) > 1e-12):
        raise ValueError('Policy uses historical Scene joint offsets; active hardware uses URDF calibration. '
                         'Collect/train in the current frame or explicitly migrate the policy; no motion started')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def joint_specs(urdf):
    result = {}
    for joint in ET.parse(urdf).findall('joint'):
        name = joint.get('name')
        if name not in JOINTS: continue
        origin, limit = joint.find('origin'), joint.find('limit')
        result[name] = dict(parent=joint.find('parent').get('link'), child=joint.find('child').get('link'),
            axis=list(map(float, joint.find('axis').get('xyz').split())),
            origin_xyz=list(map(float, origin.get('xyz', '0 0 0').split())),
            origin_rpy=list(map(float, origin.get('rpy', '0 0 0').split())),
            limits=[float(limit.get('lower')), float(limit.get('upper'))])
    if set(result) != set(JOINTS): raise ValueError('Expected the six named SO-101 URDF joints')
    return result


def raw_from_native(native, spec):
    cal = spec['calibration']; lo, hi = cal['range_min'], cal['range_max']
    if hi <= lo: raise ValueError('Invalid motor range')
    if spec['mode'] == 'DEGREES': return native * (spec['resolution'] - 1) / 360 + (lo + hi) / 2
    if spec['mode'] != 'RANGE_0_100': raise ValueError('Unsupported normalization')
    value = 100 - native if spec['apply_drive_mode'] and cal['drive_mode'] else native
    return lo + value * (hi - lo) / 100


def native_from_raw(raw, spec):
    cal = spec['calibration']; lo, hi = cal['range_min'], cal['range_max']
    if hi <= lo: raise ValueError('Invalid motor range')
    if spec['mode'] == 'DEGREES': return (raw - (lo + hi) / 2) * 360 / (spec['resolution'] - 1)
    value = (raw - lo) * 100 / (hi - lo)
    return 100 - value if spec['apply_drive_mode'] and cal['drive_mode'] else value


def unwrap(raw, anchor, period):
    delta = (raw - anchor + period / 2) % period - period / 2
    if abs(abs(delta) - period / 2) < 2: raise ValueError('Ambiguous half-turn encoder wrap; add closer references')
    return anchor + delta


def joint_raw_limits(entry):
    """Inverse URDF bounds, separate from the observed reference raw_domain."""
    return sorted(entry['anchor_raw'] + (q-entry['reference_angle_rad'])/entry['rad_per_count']
                  for q in entry['limits'])


def joint_raw_coordinate(entry, raw):
    """Resolve the encoder branch using URDF bounds, never the taught interval."""
    raw = float(raw)
    if not math.isfinite(raw): raise ValueError('Nonfinite motor position')
    lo, hi = joint_raw_limits(entry)
    period = entry['motor']['resolution']
    margin = 1e-6 / abs(entry['rad_per_count'])
    first = math.ceil((lo-margin-raw)/period)
    last = math.floor((hi+margin-raw)/period)
    if first > last: raise ValueError('Position outside URDF limits')
    if first != last: raise ValueError('Ambiguous encoder branch within URDF limits')
    return raw + first*period


def joint_angle(entry, raw):
    x = joint_raw_coordinate(entry, raw)
    return entry['rad_per_count']*(x-entry['anchor_raw']) + entry['reference_angle_rad']


def fit_joint(references, spec, joint, limits, reference_mode='angle'):
    if not references: raise ValueError('Record a reference first')
    if reference_mode not in ('angle', 'legacy_endpoints'): raise ValueError('Unknown reference mode')
    if spec['resolution']<=1 or spec['calibration']['range_max']<=spec['calibration']['range_min']:
        raise ValueError('Invalid encoder resolution/range')
    for ref in references:
        if ref['joint']!=joint or ref['split'] not in ('train','validation'): raise ValueError('Reference joint/split mismatch')
        if not np.isfinite([ref['raw'], ref['angle_rad']]).all(): raise ValueError('Nonfinite reference')
        if not limits[0] <= ref['angle_rad'] <= limits[1]: raise ValueError('Reference outside URDF limit')
        if ref['calibration_hash'] != spec['calibration_hash']: raise ValueError('Motor calibration changed')
    train = [r for r in references if r['split'] == 'train']
    if not train: raise ValueError('Training references missing')
    anchor = train[0]['raw']; period = spec['resolution']
    x = np.array([unwrap(r['raw'], anchor, period) for r in train], float)
    y = np.array([r['angle_rad'] for r in train], float)
    fallback = math.pi / 180 if joint != 'gripper' else LEGACY_MAPPER.gripper_rad_per_percent
    slope = fallback * (native_from_raw(anchor + 1, spec) - native_from_raw(anchor, spec))
    status = 'OFFSET_ONLY_NEEDS_MORE_DATA'
    if len(train) >= 2:
        if np.ptp(x) < 20 or np.ptp(y) < math.radians(5): raise ValueError('References are not distinct enough')
        if np.ptp(x) >= period / 2: raise ValueError('Encoder branch ambiguous; use closer references')
        slope, intercept = np.linalg.lstsq(np.c_[x - anchor, np.ones(len(x))], y, rcond=None)[0]
        if abs(slope) < 1e-8 or not .1 <= abs(slope) / (2 * math.pi / period) <= 5:
            raise ValueError('Implausible angle scale')
        status = 'CANDIDATE_NEEDS_VALIDATION'
    else: intercept = float(y[0])
    if joint == 'gripper' and reference_mode == 'legacy_endpoints':
        labels = {r.get('reference_kind') for r in train}
        openings = [r.get('opening_m') for r in train]
        if not {'closed', 'open'} <= labels or any(v is None or not np.isfinite(v) or v < 0 for v in openings):
            status = 'NEEDS_MORE_DATA_GRIPPER_ENDPOINTS'
        elif max(openings) - min(openings) < .002: raise ValueError('Distinct measured opening endpoints required')
    errors = [float(slope * (unwrap(r['raw'], anchor, period) - anchor) + intercept - r['angle_rad']) for r in references]
    unwrapped = [unwrap(r['raw'], anchor, period) for r in references]
    if max(unwrapped) - min(unwrapped) >= period / 2: raise ValueError('Wrap branch spans half-turn; ambiguous')
    return dict(status=status, anchor_raw=anchor, reference_angle_rad=float(intercept), rad_per_count=float(slope),
        direction=1 if slope > 0 else -1, raw_domain=[float(min(unwrapped)), float(max(unwrapped))],
        zero_raw_unwrapped=float(anchor - intercept / slope), limits=list(limits),
        train_rmse_rad=float(np.sqrt(np.mean(np.square([e for r, e in zip(references, errors) if r['split'] == 'train'])))),
        errors_rad=errors, references=copy.deepcopy(references), motor=copy.deepcopy(spec),
        reference_mode=reference_mode, evidence='estimated/manual', physical_verified=False)


def validate_joint(entry, tolerance_deg=2.):
    """Recheck evidence in existing VALIDATED_MANUAL profiles, not a worker action."""
    tolerance = math.radians(float(tolerance_deg))
    if not 0 < tolerance <= math.radians(5): raise ValueError('Tolerance must be >0 and <=5 degrees')
    rebuilt = fit_joint(entry['references'], entry['motor'], entry['references'][0]['joint'], entry['limits'],
                        entry.get('reference_mode','legacy_endpoints'))
    if rebuilt['status'] != 'CANDIDATE_NEEDS_VALIDATION': return {**rebuilt, 'status': 'NEEDS_MORE_DATA'}
    held = [r for r in rebuilt['references'] if r['split'] == 'validation']
    train = [r for r in rebuilt['references'] if r['split'] == 'train']
    if not held or any(min(abs(unwrap(r['raw'], t['raw'], entry['motor']['resolution']) - t['raw']) for t in train) < 10
                       or min(abs(r['angle_rad']-t['angle_rad']) for t in train)<math.radians(2) for r in held):
        return {**rebuilt, 'status': 'NEEDS_MORE_DATA', 'reason': 'Record a separate, distinct validation pose'}
    held_errors = [abs(e) for r, e in zip(rebuilt['references'], rebuilt['errors_rad']) if r['split'] == 'validation']
    passed = max(held_errors) <= tolerance and rebuilt['train_rmse_rad'] <= tolerance
    return {**rebuilt, 'status': 'VALIDATED_MANUAL' if passed else 'NEEDS_MORE_DATA',
        'validation': dict(passed=passed, tolerance_deg=tolerance_deg, max_error_deg=math.degrees(max(held_errors)),
                           scope='held-out manual reference; HARDWARE_NOT_TESTED by automated suite')}


def review_joint(entry, tolerance_deg=.5):
    """Enable manual live review; check fit residuals, not independent accuracy."""
    tolerance = float(tolerance_deg)
    if not np.isfinite(tolerance) or not 0 < tolerance <= 5:
        raise ValueError('Tolerance must be >0 and <=5 degrees')
    rebuilt = fit_joint(entry['references'], entry['motor'], entry['references'][0]['joint'], entry['limits'],
                        entry.get('reference_mode','legacy_endpoints'))
    errors = [abs(math.degrees(e)) for r,e in zip(rebuilt['references'], rebuilt['errors_rad']) if r['split']=='train']
    maximum = max(errors)
    supported = rebuilt['status'] in ('CANDIDATE_NEEDS_VALIDATION', 'OFFSET_ONLY_NEEDS_MORE_DATA')
    passed = supported and maximum <= tolerance
    return {**rebuilt, 'status': 'REVIEWED_MANUAL' if passed else 'NEEDS_MORE_DATA',
            'review': dict(passed=passed, tolerance_deg=tolerance, max_error_deg=maximum,
                          scope='recorded-reference fit and live display; no independent validation',
                          independent_validation=False)}


def profile_for_model(profile, model_hash):
    """Bind an in-memory copy to compatible model limits; never save or activate."""
    if profile['urdf_sha256'] == model_hash: return copy.deepcopy(profile)
    from ..so101_robot_spec import urdf_bytes, model_without_limits
    old = urdf_bytes(profile['urdf_sha256']); new = urdf_bytes(model_hash)
    if model_without_limits(old) != model_without_limits(new):
        raise ValueError('Precision profile URDF identity conflict: geometry/origin/axis/topology changed')
    before = {j.get('name'): j.find('limit') for j in ET.fromstring(old).findall('joint')}
    after = {j.get('name'): j.find('limit') for j in ET.fromstring(new).findall('joint')}
    for name in JOINTS:
        bounds = [float(after[name].get(k)) for k in ('lower','upper')]
        if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(name + ': invalid current URDF limits')
    result = copy.deepcopy(profile)
    for name, entry in result['joints'].items():
        if entry['limits'] != [float(before[name].get(k)) for k in ('lower','upper')]:
            raise ValueError(name + ': saved profile/model limits disagree')
        entry['limits'] = [float(after[name].get(k)) for k in ('lower','upper')]
    result['urdf_sha256'] = model_hash
    result['runtime_model_binding'] = {
        'saved_profile_sha256': fingerprint(profile), 'saved_urdf_sha256': profile['urdf_sha256'],
        'change': 'lower_upper_limits_only', 'persisted': False}
    return result


class PrecisionMapper:
    """Partial profiles retain legacy mapping on untouched joints, explicitly labelled."""
    def __init__(self, profile):
        self.profile = copy.deepcopy(profile)
        if profile.get('schema') != SCHEMA: raise ValueError('Unknown precision profile')
        if not profile.get('joints'):raise ValueError('Profile must contain a validated joint')
        for joint, entry in profile['joints'].items():
            if joint not in JOINTS or entry['status'] not in ('VALIDATED_MANUAL', 'REVIEWED_MANUAL'): raise ValueError('Unvalidated profile joint')
            if any(r['joint']!=joint for r in entry['references']):raise ValueError('Profile joint/reference mismatch')
            if entry['status']=='REVIEWED_MANUAL':
                checked = review_joint(entry, entry['review']['tolerance_deg'])
                if entry['review'] != checked['review']: raise ValueError('Manual review evidence differs')
            else: checked = validate_joint(entry, entry['validation']['tolerance_deg'])
            if checked['status'] != entry['status']: raise ValueError('Validation no longer passes')
            for key in ('anchor_raw', 'reference_angle_rad', 'rad_per_count', 'raw_domain'):
                if entry[key] != checked[key]: raise ValueError('Profile coefficients disagree with reference evidence')

    def _forward(self, raw, joint):
        entry = self.profile['joints'][joint]
        try: return joint_angle(entry, raw)
        except ValueError as exc: raise ValueError(joint + ': ' + str(exc)) from exc

    def raw_to_urdf(self, raw, native):
        q = LEGACY_MAPPER.calibrated_to_urdf(native)
        for joint in self.profile['joints']: q[joint] = self._forward(raw[joint], joint)
        return q

    def calibrated_to_urdf(self, native):
        q = LEGACY_MAPPER.calibrated_to_urdf(native)
        for joint, entry in self.profile['joints'].items(): q[joint] = self._forward(raw_from_native(native[joint], entry['motor']), joint)
        return q

    def calibrated_to_observation(self, native, raw=None):
        """Decode observations/leader requests beyond model limits; never authorize a goal."""
        q = LEGACY_MAPPER.calibrated_to_urdf(native)
        if raw is not None and (set(raw)!=set(JOINTS) or not np.isfinite(list(raw.values())).all()):
            raise ValueError('Invalid raw feedback packet')
        for joint, entry in self.profile['joints'].items():
            position = raw_from_native(native[joint], entry['motor']) if raw is None else float(raw[joint])
            if raw is not None:
                spec=entry['motor']; cal=spec['calibration']
                normalized = native_from_raw(np.clip(position,cal['range_min'],cal['range_max'])
                    if spec['mode']=='RANGE_0_100' else position, spec)
                if abs(normalized-native[joint])>1e-6: raise ValueError(joint+': raw/native packet mismatch')
            try: q[joint] = self._forward(position, joint)
            except ValueError as exc:
                if 'Position outside URDF limits' not in str(exc): raise
                # Out-of-model measurements on the reference encoder branch are
                # evidence, not permission to send an out-of-limit motor goal.
                x = unwrap(position, entry['anchor_raw'], entry['motor']['resolution'])
                q[joint] = entry['rad_per_count']*(x-entry['anchor_raw']) + entry['reference_angle_rad']
        return q

    def command_limits(self, bounds):
        """Representable integer goals for Teleop; never change measured angles.

        Keep the inverse strict for policy/replay. Reject disconnected or ambiguous
        encoder intervals rather than treating a gap as a commandable range.
        """
        result = dict(bounds)
        for joint, entry in self.profile['joints'].items():
            low, high = max(bounds[joint][0], entry['limits'][0]), min(bounds[joint][1], entry['limits'][1])
            if low > high: raise ValueError(joint + ': no common model/motor command range')
            slope = entry['rad_per_count']; anchor = entry['anchor_raw']; zero = entry['reference_angle_rad']
            lo, hi = sorted(anchor + (q-zero)/slope for q in (low, high))
            cal = entry['motor']['calibration']; period = entry['motor']['resolution']
            if cal['range_max']-cal['range_min'] >= period:
                raise ValueError(joint + ': ambiguous command encoder branch')
            intervals = []
            for branch in range(math.ceil((cal['range_min']-hi)/period), math.floor((cal['range_max']-lo)/period)+1):
                first = math.ceil(max(cal['range_min'], lo+branch*period))
                last = math.floor(min(cal['range_max'], hi+branch*period))
                if first > last: continue
                # Reuse the same branch checks as forward/inverse, including wraps.
                angles = sorted(joint_angle(entry, raw) for raw in (first, last))
                intervals.append(angles)
            if len(intervals) != 1:
                raise ValueError(joint + ': no single unambiguous model/motor command range')
            result[joint] = intervals[0]
        return result

    def urdf_to_calibrated(self, q):
        native = LEGACY_MAPPER.urdf_to_calibrated(q)
        for joint, entry in self.profile['joints'].items():
            if not entry['limits'][0] <= q[joint] <= entry['limits'][1]: raise ValueError(joint + ': command outside URDF limits')
            x = entry['anchor_raw'] + (q[joint] - entry['reference_angle_rad']) / entry['rad_per_count']
            cal = entry['motor']['calibration']; period = entry['motor']['resolution']
            # Float round-trip at an integer boundary is not a different branch.
            first = math.ceil((cal['range_min']-x-1e-7)/period)
            last = math.floor((cal['range_max']-x+1e-7)/period)
            if first > last:
                raise ValueError(
                    f'{joint}: command outside calibrated motor range; '
                    f'requested={math.degrees(q[joint]):+.2f} deg, '
                    f'reference-branch raw={x:.2f}, '
                    f'calibrated raw=[{cal["range_min"]}, {cal["range_max"]}]; '
                    'no goal sent. Move Leader toward its calibrated neutral pose before Start.'
                )
            if first != last: raise ValueError(joint + ': ambiguous command encoder branch')
            # Quantize at the URDF boundary without rounding a valid goal beyond it.
            lo, hi = joint_raw_limits(entry)
            lower = math.ceil(max(cal['range_min'], lo+first*period))
            upper = math.floor(min(cal['range_max'], hi+first*period))
            if lower > upper: raise ValueError(joint + ': no representable motor goal inside URDF limits')
            raw = min(upper, max(lower, round(x+first*period)))
            resolved = joint_raw_coordinate(entry, raw)
            if abs(resolved-(raw-first*period)) > 1e-6:
                raise ValueError(joint + ': command encoder branch does not roundtrip')
            native[joint] = native_from_raw(raw, entry['motor'])
        return native

    def metadata(self):
        spec = {'schema': MAPPING, 'profile': self.profile, 'joint_order': list(JOINTS),
            'input': 'original LeRobot degrees/percent; homed Present_Position before normalization',
            'output': 'urdf_joint_radian', 'evidence': 'partial_manual_calibration', 'physical_verified': False,
            'operational_range': 'URDF limits; goals also require calibrated motor range and an unambiguous encoder branch',
            'legacy_joints': [n for n in JOINTS if n not in self.profile['joints']]}
        return {**spec, 'sha256': fingerprint(spec)}


def mapper_from_metadata(metadata):
    if metadata.get('schema') != MAPPING:
        from soarm101_lab.so101_joint_mapping import DEFAULT_MAPPER
        return DEFAULT_MAPPER
    spec = {k: v for k, v in metadata.items() if k != 'sha256'}
    if fingerprint(spec) != metadata.get('sha256'): raise ValueError('Precision mapping hash mismatch')
    return PrecisionMapper(metadata['profile'])


class ProfileStore:
    def __init__(self, root): self.root = Path(root)
    def active(self, role):
        pointer = self.root / role / 'active.json'
        if not pointer.exists(): return None
        record = json.loads(pointer.read_text())
        if Path(record['id']).name!=record['id']:raise ValueError('Invalid profile ID')
        profile = json.loads((self.root / role / 'profiles' / (record['id'] + '.json')).read_text())
        if fingerprint(profile) != record['sha256']: raise ValueError('Active profile changed on disk')
        PrecisionMapper(profile)
        return profile
    def save(self, role, identity, urdf_hash, joints, provenance=None):
        directory = self.root / role / 'profiles'
        active = self.active(role)
        paths = [directory/(active['id']+'.json')] if active else []
        paths.extend(path for path in sorted(directory.glob('profile_*.json')) if path not in paths)
        for path in paths:
            try: saved = json.loads(path.read_text())
            except (OSError,ValueError): continue
            if (saved.get('schema')==SCHEMA and saved.get('id')==path.stem and saved.get('role')==role and
                saved.get('identity')==identity and saved.get('urdf_sha256')==urdf_hash and saved.get('joints')==joints and
                saved.get('physical_verified') is False and
                (provenance is None or saved.get('provenance')==provenance)):
                return str(path)
        profile = dict(schema=SCHEMA, id='profile_' + uuid.uuid4().hex, role=role, identity=identity,
                       urdf_sha256=urdf_hash, joints=copy.deepcopy(joints), created_ns=time.time_ns(), physical_verified=False)
        if provenance is not None: profile['provenance'] = copy.deepcopy(provenance)
        path = self.root / role / 'profiles' / (profile['id'] + '.json')
        atomic(path, profile)
        return str(path)
    def activate(self, role, path, identity, urdf_hash, approved=False):
        if not approved: raise ValueError('Explicit user approval required for software activation')
        path = Path(path).resolve()
        if path.parent != (self.root / role / 'profiles').resolve(): raise ValueError('Select a saved candidate of this robot')
        profile = json.loads(path.read_text()); PrecisionMapper(profile)
        if profile['id']!=path.stem:raise ValueError('Profile ID/file name mismatch')
        if profile['role'] != role or profile['identity'] != identity:
            raise ValueError('Device/calibration/model identity differs; reconnect and record references')
        effective = profile_for_model(profile, urdf_hash)
        active = self.active(role)
        if active:
            if active['identity'] != identity: raise ValueError('Active profile belongs to a different device')
            active = profile_for_model(active, urdf_hash)
            if active['id']==profile['id']: return active
            joints = {**active['joints'], **effective['joints']}
            if joints==active['joints']: return active
            if joints != effective['joints']:
                path = Path(self.save(role,identity,urdf_hash,joints,
                    provenance={'merged_active_profile':active['id'],'requested_profile':profile['id']}))
                profile = json.loads(path.read_text()); PrecisionMapper(profile)
                if profile['id']==active['id']: return active
        pointer = self.root / role / 'active.json'
        previous = json.loads(pointer.read_text()) if pointer.exists() else None
        atomic(pointer, dict(id=profile['id'], sha256=fingerprint(profile), previous=previous, approval_ns=time.time_ns()))
        return profile
    def restore(self, role, approved=False):
        if not approved: raise ValueError('Explicit user approval required')
        pointer = self.root / role / 'active.json'
        record = json.loads(pointer.read_text())
        if record['previous'] is None: pointer.unlink()
        else: atomic(pointer, record['previous'])
        return self.active(role)


class Calibration:
    """One worker owns the buses. No motor commands or register calibration writes."""
    def __init__(self, root, urdf, factory=None):
        self.base_root = Path(root); self.store = ProfileStore(root); self.urdf = Path(urdf); self.specs = joint_specs(urdf)
        self.urdf_hash = hashlib.sha256(self.urdf.read_bytes()).hexdigest()
        self.factory = factory; self.devices = {}; self.samples = {}; self.references = {}; self.exclusions = {}; self.candidates = {}; self.errors = {}; self.sessions = {}; self.loaded_profiles = {}
        self.role = 'follower'; self.joint = JOINTS[0]; self.preview_resources = None
    def connect(self, payload):
        role = payload['role']
        if role not in ('leader', 'follower'): raise ValueError('Unknown robot')
        if role in self.devices: raise ValueError('Disconnect this calibration device first')
        fake = bool(payload.get('fake'))
        if self.devices and any(isinstance(d,FakeCalibrationDevice)!=fake for d in self.devices.values()):
            raise ValueError('Do not mix Fake and real devices in one calibration session')
        self.store = ProfileStore(self.base_root/'fake' if fake else self.base_root)
        if payload.get('fake'): device = FakeCalibrationDevice(role)
        else:
            if self.factory: device = self.factory(role, payload['hardware'])
            else:
                from .hardware import CalibrationDevice
                device = CalibrationDevice(role, payload['hardware'])
        try: device.connect()
        except BaseException:
            device.close(); raise
        self.devices[role] = device; self.role = role
        self.sessions[role] = 'references_' + uuid.uuid4().hex
        self.references[role] = {}; self.exclusions[role] = []; self.candidates[role] = {}; self.errors.pop(role, None)
        self.loaded_profiles[role] = None
        try:active = self.store.active(role)
        except (ValueError,OSError,KeyError):active = None  # Invalid pointers remain available for explicit restore.
        if active and active['identity']==device.identity:
            profile_for_model(active, self.urdf_hash)
            self.loaded_profiles[role] = active['id']
            for joint,entry in active['joints'].items():
                self.references[role][joint] = copy.deepcopy(entry['references'])
                for ref in self.references[role][joint]:
                    ref.setdefault('reference_id',uuid.uuid4().hex)
        return 'Read-only connected; no torque/goal/calibration register changed'
    def active_references(self, role, joint):
        excluded = {item['reference_id'] for item in self.exclusions.get(role, [])}
        return [ref for ref in self.references.get(role, {}).get(joint, []) if ref['reference_id'] not in excluded]
    def save_references(self, role, references=None, exclusions=None):
        # Original measurements stay in the journal. Exclusions only affect fit/UI.
        atomic(self.store.root/role/'references'/(self.sessions[role]+'.json'),
               {'schema':'so101.joint-references/1','identity':self.devices[role].identity,'urdf_sha256':self.urdf_hash,
                'references':self.references[role] if references is None else references,
                'exclusions':self.exclusions[role] if exclusions is None else exclusions,
                'loaded_profile_id':self.loaded_profiles.get(role)})
    def tick(self):
        if self.role not in self.devices: return
        try:
            device = self.devices[self.role]; sample = device.sample(self.joint, live=True)
            profile_error=None
            try:active = self.store.active(self.role)
            except ValueError as exc:active=None;profile_error=str(exc)
            if active:
                try: active = profile_for_model(active, self.urdf_hash)
                except ValueError as exc: active=None;profile_error=str(exc)
            entry = active['joints'].get(self.joint) if active else None
            legacy = LEGACY_MAPPER.calibrated_to_urdf({n: sample['legacy_position'] if n == self.joint else 0. for n in JOINTS})[self.joint]
            conversion_error = profile_error
            try: converted = None if profile_error else PrecisionMapper(active)._forward(sample['raw'], self.joint) if entry else legacy
            except ValueError as exc: converted = None; conversion_error = str(exc)
            self.samples[self.role] = {**sample, 'urdf_angle_rad': converted,
                'legacy_urdf_angle_rad': legacy,
                'estimated_zero_error_rad': converted - legacy if converted is not None else None,
                'zero_error_scope': 'estimated correction versus legacy mapping; not an external measurement',
                'torque_status': 'RELEASED' if sample['torque']==0 else 'TORQUED',
                'calibration_status': 'MAPPING_ERROR' if conversion_error else entry['status'] if entry else 'legacy_unverified'}
            if conversion_error: self.errors[self.role]=conversion_error
            else: self.errors.pop(self.role, None)
        except Exception as exc: self.errors[self.role] = str(exc); self.samples.pop(self.role, None)
    def status(self):
        robots={}
        for role in ('follower','leader'):
            try:
                active=self.store.active(role)
                if active: active=profile_for_model(active,self.urdf_hash)
            except (ValueError,OSError,KeyError) as exc:active=None;self.errors[role]=str(exc)
            robots[role]=dict(connected=role in self.devices,identity=self.devices[role].identity if role in self.devices else None,
                active_profile=active,active_pointer_exists=(self.store.root/role/'active.json').exists(),
                sample=self.samples.get(role),references={joint:self.active_references(role,joint) for joint in self.references.get(role,{})},
                excluded_references=self.exclusions.get(role,[]),candidates=self.candidates.get(role,{}))
        return dict(selected_role=self.role, selected_joint=self.joint, model=self.specs, errors=self.errors,
            profile_root=str(self.store.root), hardware_status='HARDWARE_NOT_TESTED' if any(isinstance(d,FakeCalibrationDevice) for d in self.devices.values()) else 'READ_ONLY_DEVICE_SESSION',
            robots=robots)
    def command(self, name, payload):
        role = payload.get('role', self.role); joint = payload.get('joint', self.joint)
        if role not in ('leader', 'follower') or joint not in JOINTS: raise ValueError('Unknown robot/joint')
        if name == 'connect': return self.connect(payload)
        if name == 'select': self.role, self.joint = role, joint; self.samples.pop(role, None); return 'Selected; no motor write'
        if name == 'disconnect':
            if role in self.devices: self.devices.pop(role).close()
            self.samples.pop(role, None); return 'Disconnected'
        if role not in self.devices: raise ValueError('Connect selected robot first')
        device = self.devices[role]
        if name == 'fake_raw':
            if not isinstance(device, FakeCalibrationDevice): raise ValueError('Fake input unavailable on hardware')
            device.raw[joint] = int(payload['raw']); return 'Fake feedback changed (no hardware)'
        if name == 'release':
            if payload.get('supported_and_approved') is not True: raise ValueError('Support robot and explicitly approve torque release')
            device.release(joint)
            if device.sample(joint)['torque'] != 0: raise ValueError('Torque release readback failed')
            return 'Selected joint torque released; support against gravity'
        if name == 'record':
            sample = device.sample(joint)
            if sample['torque'] != 0: raise ValueError('Release torque and support robot before recording')
            if abs(device.sample(joint)['raw'] - sample['raw']) > 3: raise ValueError('Joint moved during measurement')
            if (payload.get('split', 'train') != 'train' or payload.get('reference_kind', 'angle') != 'angle'
                    or 'opening_m' in payload):
                raise ValueError('Record only angle references for manual review')
            ref = {**sample, 'angle_rad': float(payload['angle_rad']), 'split': 'train',
                   'reference_kind': 'angle',
                   'evidence': 'estimated/manual', 'reference_id': uuid.uuid4().hex}
            if not self.specs[joint]['limits'][0] <= ref['angle_rad'] <= self.specs[joint]['limits'][1]: raise ValueError('Reference exceeds URDF limits')
            references = copy.deepcopy(self.references[role]); references.setdefault(joint, []).append(ref)
            self.save_references(role, references=references)
            self.references[role] = references
            self.candidates[role].pop(joint, None); return ref
        if name == 'exclude_reference':
            if payload.get('approved') is not True: raise ValueError('Explicit approval required to exclude a reference')
            identifier = payload.get('reference_id')
            if not any(ref['reference_id']==identifier for ref in self.active_references(role,joint)):
                raise ValueError('Reference not found in the selected robot/joint or already excluded')
            reason = str(payload.get('reason', '')).strip()
            if not reason: raise ValueError('Reference exclusion reason required')
            exclusions = self.exclusions[role] + [dict(reference_id=identifier, joint=joint, reason=reason, excluded_ns=time.time_ns())]
            self.save_references(role, exclusions=exclusions)
            self.exclusions[role] = exclusions
            self.candidates[role].pop(joint, None)
            return dict(reference_id=identifier, remaining=len(self.active_references(role,joint)),
                        scope='fit reference excluded; original measurement and active profile preserved')
        if name == 'fit':
            result = fit_joint(self.active_references(role,joint), device.motor_spec(joint), joint, self.specs[joint]['limits'])
            self.candidates[role][joint] = result; return result
        if name == 'review':
            entry = self.candidates[role][joint]; result = review_joint(entry, payload.get('tolerance_deg', .5))
            self.candidates[role][joint] = result; return result
        if name == 'save':
            active = self.store.active(role)
            if active: active = profile_for_model(active, self.urdf_hash)
            entries = copy.deepcopy(active['joints']) if active else {}
            for name,entry in self.candidates[role].items():
                if entry['status'] in ('REVIEWED_MANUAL','VALIDATED_MANUAL'): entries[name] = copy.deepcopy(entry)
            entries[joint] = self.candidates[role][joint]
            return {'path': self.store.save(role, device.identity, self.urdf_hash, entries)}
        if name == 'activate':
            if not device.unchanged(): raise ValueError('Motor register/file calibration changed; reconnect')
            if device.sample(joint)['torque'] != 0: raise ValueError('Release selected joint before software activation')
            result = self.store.activate(role, payload['path'], device.identity, self.urdf_hash, payload.get('approved') is True)
            return dict(id=result['id'], scope='software mapping only; reconnect Real/Sim to apply', registers_written=False)
        if name == 'restore': return self.store.restore(role, payload.get('approved') is True)
        if name == 'compare':
            current = device.sample(joint); candidate = self.candidates[role][joint]
            q = joint_angle(candidate, current['raw'])
            active=self.store.active(role)
            legacy=LEGACY_MAPPER.calibrated_to_urdf({n:current['legacy_position'] if n==joint else 0. for n in JOINTS})[joint]
            try:old=PrecisionMapper(profile_for_model(active, self.urdf_hash))._forward(current['raw'],joint) if active and joint in active['joints'] else legacy
            except ValueError:old=None
            return {'current_raw': current['raw'], 'candidate_rad': q,'current_urdf_rad':old,
                    'difference_rad':q-old if old is not None else None,'legacy_urdf_rad':legacy,
                    'legacy_position': current['legacy_position'], 'evidence': 'estimated/manual'}
        if name == 'compare_robots':
            if payload.get('same_pose_confirmed') is not True: raise ValueError('User must confirm same static physical pose')
            pair = {}
            for r in ('leader', 'follower'):
                d = self.devices[r]; sample = d.sample(joint); active = self.store.active(r)
                if not active or joint not in active['joints']: raise ValueError('Activate validated profile for both selected joints')
                pair[r] = {**sample, 'urdf_angle_rad': PrecisionMapper(profile_for_model(active, self.urdf_hash))._forward(sample['raw'], joint)}
            return {**pair, 'difference_rad': pair['leader']['urdf_angle_rad'] - pair['follower']['urdf_angle_rad'], 'scope': 'user-confirmed static pose; not tracking error'}
        if name == 'preview': return self.preview(role, joint, float(payload['angle_rad']), payload.get('renderer', 'egl'))
        raise ValueError('Unsupported calibration command')
    def preview(self, role, joint, angle, renderer):
        if not self.specs[joint]['limits'][0] <= angle <= self.specs[joint]['limits'][1]: raise ValueError('Reference exceeds URDF limits')
        from tempfile import TemporaryDirectory
        from ..simulation.mjcf import robot_xml
        from ..simulation.mujoco_backend import configure_rendering
        configure_rendering(renderer)
        import mujoco
        if self.preview_resources is None: self.preview_resources = TemporaryDirectory(prefix='joint_preview_')
        root = Path(self.preview_resources.name); xml, _, _ = robot_xml(self.urdf, np.eye(4))
        ET.ElementTree(xml).write(root / 'robot.xml')
        (root / 'scene.xml').write_text('<mujoco><include file="robot.xml"/><visual><global offwidth="640" offheight="480"/></visual><worldbody><light pos="0 0 1"/></worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_path(str(root / 'scene.xml')); data = mujoco.MjData(model); device = self.devices[role]
        native = {}; raw = {}
        for n in JOINTS:
            sample = device.sample(n); native[n] = sample['legacy_position']; raw[n] = sample['raw']
        active = self.store.active(role); mapper = PrecisionMapper(profile_for_model(active, self.urdf_hash)) if active else LEGACY_MAPPER
        common = mapper.raw_to_urdf(raw, native) if active else mapper.calibrated_to_urdf(native)
        camera = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, camera); camera.distance = .65; camera.azimuth = 135; camera.elevation = -25
        from PIL import Image
        with mujoco.Renderer(model, height=480, width=640) as view:
            for label in ('current', 'reference'):
                for n in JOINTS: data.qpos[model.joint(n).qposadr[0]] = angle if label == 'reference' and n == joint else common[n]
                mujoco.mj_forward(model, data); view.update_scene(data, camera=camera)
                self.store.root.mkdir(parents=True, exist_ok=True)
                Image.fromarray(view.render()).save(self.store.root / (label + '.png'))
        return {'current': str(self.store.root / 'current.png'), 'reference': str(self.store.root / 'reference.png'), 'hardware_commanded': False}
    def close(self):
        for device in self.devices.values(): device.close()
        self.devices.clear()
        if self.preview_resources: self.preview_resources.cleanup(); self.preview_resources = None


class FakeCalibrationDevice:
    """Explicit test mode, never usable with a real device-bound profile."""
    def __init__(self, role):
        self.identity = {'role': role, 'port': 'FAKE', 'id': 'FAKE_' + role, 'calibration_sha256': 'fake'}
        self.raw = {n: 2048 for n in JOINTS}; self.torque = {n: 1 for n in JOINTS}; self.writes = []
    def connect(self): pass
    def unchanged(self): return True
    def motor_spec(self, joint):
        return dict(calibration_hash='fake', resolution=4096, mode='RANGE_0_100' if joint == 'gripper' else 'DEGREES',
            apply_drive_mode=False, calibration=dict(id=JOINTS.index(joint) + 1, homing_offset=37, drive_mode=0, range_min=0, range_max=4095))
    def sample(self, joint, live=False):
        return dict(raw=self.raw[joint], legacy_position=native_from_raw(self.raw[joint], self.motor_spec(joint)),
                    torque=self.torque[joint], timestamp_ns=time.monotonic_ns(), device_identity=self.identity,
                    joint=joint, calibration_hash='fake', motor=self.motor_spec(joint))
    def release(self, joint): self.torque[joint] = 0; self.writes.append(('Torque_Enable', joint, 0))
    def close(self): pass
