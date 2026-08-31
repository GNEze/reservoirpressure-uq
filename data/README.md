# Data

Place the raw Volve production record here as:

```
data/Volve production data.xlsx
```

Source: Equinor ASA (2018), Volve field data set, open data release —
https://www.equinor.com/energy/volve-data-sharing

This file is not redistributed in this repository (see `.gitignore`); download it
from the source above. `01_volve_data_exploration.py` reads the `Daily Production
Data` sheet and expects the column names as published (`NPD_WELL_BORE_NAME`,
`DATEPRD`, `AVG_DOWNHOLE_PRESSURE`, `BORE_OIL_VOL`, `BORE_WAT_VOL`, `BORE_GAS_VOL`).
