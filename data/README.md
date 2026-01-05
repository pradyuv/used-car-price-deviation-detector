pct_blank = df["clean_title"].isna().mean() * 100 -> 14.9 percent blank
We aren't going to drop those rows or coerce to False, we would lose/penalize a lot of data
Treat blanks as unknown

Dropped engine and transmission features in favor of deriving simpler features from which to train my model
--> engine_displacement_litres and transmission_type
Engine and tranmission had long strings, which were not normalized


As of Jan 5 2026:
Validation MAE: 30,879
Validation RMSE: 130,883

Investigating why....:

 Extreme listings (top residuals in validation set)
  Here are the 10 largest absolute residuals from the validation split:

   listed_price  expected_price     residual       brand                   model
  model_year  milage fuel_type  engine_displacement_liters transmission_type
  clean_title  accident
        2954083    41655.173816 2.912428e+06    maserati       quattroporte base
  2005 32000.0  gasoline                         4.2           unknown
  yes         1
        1950995    49440.723645 1.901554e+06     bugatti veyron 16.4 grand sport
  2011  6330.0  gasoline                         8.0         automatic
  yes         0
         417500    52964.906334 3.645351e+05     porsche              911 gt2 rs
  2018  4529.0  gasoline                         3.8         automatic
  yes         0
         319900    49473.864319 2.704261e+05     ferrari         488 spider base
  2019  9005.0  gasoline                         3.9         automatic
  unknown         0
         314900    54647.528173 2.602525e+05 rolls-royce              ghost base
  2021   850.0  gasoline                         6.7         automatic
  yes         0
         279000    48531.532639 2.304685e+05    maserati               mc20 base
  2022  1087.0  gasoline                         3.0         automatic
  yes         0
         279950    53842.871257 2.261071e+05     porsche                 911 gt3
  2023   265.0  gasoline                         4.0            manual
  yes         0
         259500    51354.843741 2.081452e+05     ferrari               roma base
  2022  2250.0  gasoline                         3.9         automatic
  yes         0
         239995    56336.911089 1.836581e+05     bentley              bentayga s
  2023  2600.0  gasoline                         4.0         automatic
  yes         0
         215000    46808.487138 1.681915e+05     ferrari             gtc4lusso t
  2019  8870.0  gasoline                         3.9         automatic
  yes         0

  These are all high‑end/exotic listings. The model is predicting “typical car”
  prices for them because those categories are extremely rare. That’s why the
  residuals are huge and why RMSE is inflated.


I’m keeping these outliers and accepting a high RMSE because they are
legitimate listings and extremely rare. The model uses brand/model, but for
exotics there’s too little data to form stable local averages, so
predictions regress toward typical car prices. This inflates RMSE but
doesn’t materially affect stability for the bulk of listings, which is the
goal of the system.