# data/schemas/

Data contracts — one file per dataset describing column names, types, units, and expected ranges/nullability.

Update the relevant schema file whenever a raw data source's shape changes (new column, renamed field, changed units, etc.) so the other two teammates aren't surprised downstream in `features/` or `models/`. A simple markdown table or a `.json`/`.yaml` schema per dataset is fine — consistency matters more than format.
