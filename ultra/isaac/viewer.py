"""Viewer helpers for the ULTRA environments: camera follow, debug drawing and frame capture.

All positions passed in are env-local (the legacy ULTRA convention); the environment origin is added here.
"""

import os

import numpy as np


class DebugSphere:
    """Replacement for ``gymutil.WireframeSphere``: drawn as a single point whose size follows the radius."""

    def __init__(self, radius=0.02, color=(1.0, 1.0, 1.0)):
        self.radius = radius
        self.color = tuple(color)


class UltraViewer:
    POINT_SIZE_PER_METER = 400.0

    def __init__(self, env, capture_resolution=(1280, 720)):
        self.env = env
        self._draw = None
        self._capture_resolution = capture_resolution
        self._rgb_annotator = None
        self._lines = ([], [], [], [])
        self._points = ([], [], [])

    # -- camera -------------------------------------------------------------------------------------------------------
    def look_at(self, env_id, eye, target):
        origin = self._origin(env_id)
        self.env.sim.set_camera_view(tuple(origin + np.asarray(eye)), tuple(origin + np.asarray(target)))

    # -- debug drawing ------------------------------------------------------------------------------------------------
    def _interface(self):
        if self._draw is None:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("isaacsim.util.debug_draw")
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
        return self._draw

    def clear(self):
        self._lines = ([], [], [], [])
        self._points = ([], [], [])
        draw = self._interface()
        draw.clear_lines()
        draw.clear_points()

    def draw_sphere(self, env_id, pos, sphere: DebugSphere):
        p = self._origin(env_id) + np.asarray(pos, dtype=np.float64)
        self._points[0].append(tuple(p))
        self._points[1].append((*sphere.color, 1.0))
        self._points[2].append(max(1.0, sphere.radius * self.POINT_SIZE_PER_METER))

    def draw_line(self, env_id, p1, p2, color, width=2.0):
        origin = self._origin(env_id)
        self._lines[0].append(tuple(origin + np.asarray(p1, dtype=np.float64)))
        self._lines[1].append(tuple(origin + np.asarray(p2, dtype=np.float64)))
        self._lines[2].append((*color, 1.0))
        self._lines[3].append(width)

    def flush(self):
        """Send everything queued since the last :meth:`clear` to the renderer."""
        draw = self._interface()
        if self._points[0]:
            draw.draw_points(*self._points)
        if self._lines[0]:
            draw.draw_lines(*self._lines)

    # -- frame capture ------------------------------------------------------------------------------------------------
    def capture_frame(self):
        """RGB image (H, W, 3) of the viewport camera, or None while the renderer warms up."""
        if self._rgb_annotator is None:
            import omni.replicator.core as rep

            render_product = rep.create.render_product("/OmniverseKit_Persp", self._capture_resolution)
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([render_product])
        data = self._rgb_annotator.get_data()
        data = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape)
        if data.size == 0:
            return None
        return data[:, :, :3].copy()

    def save_frame(self, path):
        frame = self.capture_frame()
        if frame is None:
            return False
        from PIL import Image

        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(frame).save(path)
        return True

    def _origin(self, env_id):
        return self.env.scene.env_origins[int(env_id)].detach().cpu().numpy().astype(np.float64)
