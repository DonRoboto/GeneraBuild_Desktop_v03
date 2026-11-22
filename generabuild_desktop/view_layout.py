"""Distribución de las cuatro fotografías, independiente de Qt.

Conservar un solo PhotoPanel por vista evita duplicar escenas, zoom,
listeners y anotaciones al alternar entre cuadrícula y vista única.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PhotoLayoutPlan:
    mode: str
    active: int
    positions: dict[int, tuple[int, int, int, int]]

    @property
    def visible(self) -> tuple[int, ...]:
        return tuple(self.positions)


def plan_photo_layout(mode: str, active: int) -> PhotoLayoutPlan:
    """Posiciones (fila, columna, span_filas, span_columnas) para las vistas 1..4."""
    if mode not in {'grid', 'single'}:
        raise ValueError(f'Modo de vista no reconocido: {mode!r}')
    if not isinstance(active, int) or not 1 <= active <= 4:
        raise ValueError(f'Vista activa inválida: {active!r}')
    if mode == 'grid':
        positions = {i: ((i - 1) // 2, (i - 1) % 2, 1, 1)
                     for i in range(1, 5)}
    else:
        positions = {active: (0, 0, 2, 2)}
    return PhotoLayoutPlan(mode, active, positions)
