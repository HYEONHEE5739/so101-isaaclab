from soarm101_lab.real2sim.workspaces.grasp import StableGrasp


def feed(t, i, left=True, right=True, moving=True, rel=None, quat=None):
    x = i*.002 if moving else 0
    return t.update(i, rel or [0,0,.02], quat or [1,0,0,0],
                    [x,0,0], [x,0,.02], left, right)


def test_bilateral_moving_hold_and_historical_boundary():
    t=StableGrasp(.05,[.024]*3)
    for i in range(4):
        assert not feed(t,i)
    assert feed(t,4)
    assert feed(t,5,right=False)
    assert not t.details['stable']


def test_one_finger_stationary_and_brief_contact_rejected():
    for moving,left,right in [(True,True,False),(True,False,True),(False,True,True)]:
        t=StableGrasp(.05,[.024]*3)
        for i in range(20):
            assert not feed(t,i,left,right,moving)
    t=StableGrasp(.05,[.024]*3)
    for i in range(20):
        assert not feed(t,i,right=i%3!=0)


def test_slip_rotation_duplicate_calls_and_fresh_episode():
    t=StableGrasp(.05,[.024]*3)
    for i in range(20):
        assert not feed(t,i,rel=[i*.004,0,0])
    t=StableGrasp(.05,[.024]*3)
    for i in range(20):
        assert not feed(t,i,quat=[1,0,0,0] if i%2 else [.707,0,0,.707])
    t=StableGrasp(.05,[.024]*3)
    for _ in range(20):
        assert not feed(t,0)
    assert not feed(StableGrasp(.05,[.024]*3),0)
