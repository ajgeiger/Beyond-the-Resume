# data/raw

The source documents the index is built from. They are copied here by
`make sources` from the repositories listed in `configs/sources.toml` and are
not committed. Only this README is tracked.

| File | Copied from |
| --- | --- |
| resume.pdf | resume/Alexander_Geiger_Resume.pdf |
| resume_build.py | resume/build.py |
| portfolio_site_resume.py | topic-explorer-dashboard/config/Resume.py |
| portfolio_site_readme.md | topic-explorer-dashboard/README.md |
| headline_radar_readme.md | headline-radar/README.md |
| portfolio_site_home.py | topic-explorer-dashboard/pages/HomePage.py |
| portfolio_site_projects.py | topic-explorer-dashboard/pages/ProjectsPage.py |
| portfolio_site_projects_registry.py | topic-explorer-dashboard/portfolio/ProjectRegistry.py |
| portfolio_site_pages.md | rag-chatbot/docs/portfolio_site_pages.md |
| natural_gas_forecast_readme.md | natural-gas-forecasting/README.md |

The built index in `data/processed/index` is committed, so serving and
deploying never need these files. Contact details are removed before
anything is embedded.
