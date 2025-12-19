pct_blank = df["clean_title"].isna().mean() * 100 -> 14.9 percent blank
We aren't going to drop those rows or coerce to False, we would lose/penalize a lot of data
Treat blanks as unknown

Dropped engine and transmission features in favor of deriving simpler features from which to train my model
--> engine_displacement_litres and transmission_type
Engine and tranmission had long strings, which were not normalized

