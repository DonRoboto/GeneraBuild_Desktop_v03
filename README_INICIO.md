# GeneraBuild Studio — Inicio rápido

**Para reproducir el escenario de Guanajuato 8**, abre GeneraBuild Studio y carga el proyecto del sitio. En las capturas el directorio se llama `workspace/sites/gto_v5`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Para generar imágenes, también se requiere PyTorch CUDA y requirements-gpu.txt.
python -m generabuild_desktop
```

1. **File → Open project folder...** → `workspace/sites/gto_v5` (o `gta8_v5`).
2. **Scenario → Target floors** → **8, 9 o 10** pisos totales.
3. Conserva **Vertical extension (3→N)** y **Assumed height per floor = 3.00 m**.
4. Pulsa **Prepare scenario · CPU**.
5. Pulsa **GENERAR · Stable Diffusion + ControlNet (GPU)**.
6. Consulta el resultado en **Results**.

La configuración documentada para el escenario de 8 pisos utiliza **Seed 1234**, **Steps 40**, **CFG 5.50** y **ControlNet 0.65**. Las alturas nominales propuestas son **24 m, 27 m y 30 m**, respectivamente.

**[Leer el manual ilustrado completo](docs/MANUAL_GENERABUILD_STUDIO.md)**.
