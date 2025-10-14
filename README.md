# kelmoidAI_Genesis_llm

This monorepo unifies two projects while preserving their Git histories:
- AdvancedCAD
- MECH_MIND

Merge strategy: git subtree with per-project prefixes under the repository root.

Imported structure:
- MECH_MIND/ (imported)
- AdvancedCAD/ (imported from local snapshot; history not preserved)

Setup (Windows PowerShell):
- python -m venv venv
- venv\Scripts\Activate
- pip install -r requirements.txt

Run the Gradio app:
- python MECH_MIND/app.py

Generate a CAD demo (exports to outputs/demo_model.stl):
- python scripts/generate_cad_demo.py

Note on Git history:
- AdvancedCAD was added from the local workspace. If you want to preserve its Git history in this monorepo, provide the remote URL/branch and we can re-import using:
  git subtree add --prefix=AdvancedCAD <remote> <branch>
