# kelmoidAI_Genesis_llm

This monorepo unifies two projects while preserving their Git histories:
- AdvancedCAD
- MECH_MIND

Merge strategy: git subtree with per-project prefixes under the repository root.

Imported structure:
- MECH_MIND/ (imported via git subtree)
- AdvancedCAD/ (pending import — provide Git URL or local repo path to preserve history)

Quick start (demo CAD export):
- Ensure AdvancedCAD core is available either inside this repo at `AdvancedCAD/src` or as a sibling directory `../AdvancedCAD/src`.
- Create a virtual environment and install dependencies for AdvancedCAD if not already:
  - Windows: `python -m venv venv && venv\Scripts\activate && pip install -r ../AdvancedCAD/requirements.txt`
- Run the demo:
  - `python scripts/generate_cad_demo.py`
- The STL will be written to `outputs/demo_model.stl`.

To preserve AdvancedCAD history in this monorepo:
- Provide the Git URL or local path to the AdvancedCAD repository and branch name. We will import it with `git subtree add --prefix=AdvancedCAD <remote> <branch>` preserving its full history.
