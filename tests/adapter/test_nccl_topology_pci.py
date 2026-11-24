import os
from hcu_adapter.infra import nccl_topo_guard as guard

def write_topology(path, buses):
    path.write_text("<system>" + "".join(
        f'<pci busid="{bus}"><gpu dev="{i}" gcn="gfx938"/></pci>'
        for i, bus in enumerate(buses)) + "</system>")
    return str(path)

def test_equal_gpu_count_rejects_foreign_pci_topology(tmp_path, monkeypatch):
    path = write_topology(tmp_path / "topo.xml", ["0000:09:00.0", "0000:36:00.0"])
    monkeypatch.setattr(guard, "_host_gpu_busids", lambda: {"0000:05:00.0", "0000:30:00.0"} )
    action, reason = guard._check(path, 2)
    assert action == "clear"
    assert "PCI" in reason

def test_matching_pci_topology_is_preserved(tmp_path, monkeypatch):
    buses = ["0000:05:00.0", "0000:30:00.0"]
    path = write_topology(tmp_path / "topo.xml", buses)
    monkeypatch.setattr(guard, "_host_gpu_busids", lambda: set(buses) )
    assert guard._check(path, 2)[0] == "keep"

def test_unknown_host_pci_topology_is_preserved(tmp_path, monkeypatch):
    path = write_topology(tmp_path / "topo.xml", ["0000:09:00.0"])
    monkeypatch.setattr(guard, "_host_gpu_busids", lambda: set() )
    assert guard._check(path, 1)[0] == "keep"

def test_host_busids_come_from_render_device_links(tmp_path):
    device = tmp_path / "devices" / "0000:05:00.0"
    device.mkdir(parents=True)
    render = tmp_path / "drm" / "renderD128"
    render.mkdir(parents=True)
    (render / "device").symlink_to(device, target_is_directory=True)
    assert guard._host_gpu_busids(tmp_path / "drm") == {"0000:05:00.0"}

def test_install_clears_equal_count_foreign_topology(tmp_path, monkeypatch):
    path = write_topology(tmp_path / "topo.xml", ["0000:09:00.0"])
    monkeypatch.setenv("NCCL_TOPO_FILE", path)
    monkeypatch.setattr(guard, "_visible_gpus", lambda: 1)
    monkeypatch.setattr(guard, "_host_gpu_busids", lambda: {"0000:05:00.0"} )
    guard.install()
    assert "NCCL_TOPO_FILE" not in os.environ

def test_explicit_guard_optout_keeps_foreign_topology(tmp_path, monkeypatch):
    path = write_topology(tmp_path / "topo.xml", ["0000:09:00.0"])
    monkeypatch.setenv("NCCL_TOPO_FILE", path)
    monkeypatch.setenv("HCU_NCCL_TOPO_GUARD", "0")
    monkeypatch.setattr(guard, "_visible_gpus", lambda: 1)
    monkeypatch.setattr(guard, "_host_gpu_busids", lambda: {"0000:05:00.0"} )
    guard.install()
    assert os.environ["NCCL_TOPO_FILE"] == path
