source('library.R')

past_date = as.Date('1970-1-1')
future_date = as.Date('2100-1-1')

state_abbr_name = list(
  BadenWuerttemberg = 'BW',
  Bayern = 'BY',
  Berlin = 'BE',
  Brandenburg = 'BB',
  Bremen = 'HB',
  Hamburg = 'HH',
  Hessen = 'HE',
  MecklenburgVorpommern = 'MV',
  Niedersachsen = 'HB,NI',
  NordrheinWestfalen = 'NW',
  RheinlandPfalz = 'RP',
  Saarland = 'SL',
  Sachsen = 'SN',
  SachsenAnhalt = 'ST',
  SchleswigHolstein = 'SH',
  Thueringen = 'TH'
)

event_levels = c('', 'Fog-Rain', 'Fog-Snow', 'Fog-Thunderstorm',
             'Rain-Snow-Hail-Thunderstorm', 'Rain-Snow', 'Rain-Snow-Hail',
             'Fog-Rain-Hail', 'Fog', 'Fog-Rain-Hail-Thunderstorm', 'Fog-Snow-Hail',
             'Rain-Hail', 'Rain-Hail-Thunderstorm', 'Fog-Rain-Snow', 'Rain-Thunderstorm',
             'Fog-Rain-Snow-Hail', 'Rain', 'Thunderstorm', 'Snow-Hail',
             'Rain-Snow-Thunderstorm', 'Snow', 'Fog-Rain-Thunderstorm')

min_max_scaler <- function(X){
  (X-min(X))/(max(X) - min(X))
}
parse_weather <- function(weather_csv_dir){
  filenames = list.files(weather_csv_dir, pattern='*.csv$')
  data = list()
  for(f in filenames){
    state_name = substr(f, 0, nchar(f)-4)
    state_code = state_abbr_name[state_name]
    weather_data = fread(paste(weather_csv_dir, f, sep='/'))
    weather_data[,State:=state_code]
    data = list.append(data, weather_data)
  }
  data = rbindlist(data)
  
  temperature_cols = c('Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC')
  data[, c(temperature_cols):=lapply(.SD, function(x){(x-10)/30}), 
       .SDcols = temperature_cols]
  
  humidity_cols = c('Max_Humidity', 'Mean_Humidity', 'Min_Humidity')
  data[, c(humidity_cols):=lapply(.SD, function(x){(x-50)/50}), 
       .SDcols = humidity_cols]
  
  data[, Max_Wind_SpeedKm_h:=Max_Wind_SpeedKm_h/50]
  data[, Mean_Wind_SpeedKm_h:=Mean_Wind_SpeedKm_h/30]
  
  data[is.na(CloudCover), CloudCover:=0]
  
  data[,Events:=as.integer(factor(Events, levels=event_levels))]
  
  weather_features = c('State', 'Date', temperature_cols, humidity_cols, 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h',
                       'CloudCover', 'Events')
  
  data[, Date:=as.Date(Date)]
  data[, weather_features, with=FALSE]
}


parse_googletrend<- function(googletrend_dir){
  filenames = list.files(googletrend_dir, pattern='*.csv$')
  data = list()
  i = 1
  for(f in filenames){
    state_code = substr(f, nchar(f)-5, nchar(f)-4)
    if(state_code=='NI'){
      state_code = 'HB,NI'
    }
    x = fread(paste(googletrend_dir, f, sep='/'))
    x[, State:=state_code]
    data = list.append(data, x)
  }
  
  trend_data = rbindlist(data)
  trend_data[, Trend_Val:=trend_data[[2]]/100]
  
  n = nrow(trend_data)
  
  end_dates = character(n)
  
  
  date_range = strsplit(trend_data[[1]], ' - ', fixed=TRUE)
  for(i in 1:n){
    end_dates[i] = unlist(date_range[i])[2]
  }
  trend_data[, Date:=as.Date(end_dates)]
  trend_data[, Year:=year(Date)]
  trend_data[, WeekOfYear:=week(Date)]
  
  trend_data_missing = CJ(State=unique(trend_data$State), Year=c(2013,2014), WeekOfYear=53, Trend_Val=0)
  trend_data = rbindlist(list(trend_data, trend_data_missing), fill=T)
  setkey(trend_data, 'Year', 'WeekOfYear')
  trend_data[, Trend_Val:=ifelse(Trend_Val==0, rollsum(Trend_Val, 3, na.pad=T )/2, Trend_Val), by='State']
  trend_data[, .(State, Year, WeekOfYear, Trend_Val)]
}

add_bw_features <- function(time_window){
  # Promo
  all_data[, Tmp_Date:=cummax(ifelse(Promo!=0, Date, past_date)), by=Store]
  all_data[, Tmp_Date:=as.Date(Tmp_Date)]
  all_data[, Promo_Backward:=as.integer(Date-Tmp_Date)]
  all_data[Promo_Backward>time_window, Promo_Backward:=time_window]
  # SchoolHoliday
  all_data[, Tmp_Date:=cummax(ifelse(SchoolHoliday!=0, Date, past_date)), by=Store]
  all_data[, Tmp_Date:=as.Date(Tmp_Date)]
  all_data[, SchoolHoliday_Backward:=as.integer(Date-Tmp_Date)]
  all_data[SchoolHoliday_Backward>time_window, SchoolHoliday_Backward:=time_window]
  
  # StateHoliday
  all_data[, Tmp_Date:=cummax(ifelse(StateHolidayN!=0, Date, past_date)), by=Store]
  all_data[, Tmp_Date:=as.Date(Tmp_Date)]
  all_data[, StateHoliday_Backward:=as.integer(Date-Tmp_Date)]
  all_data[StateHoliday_Backward>time_window, StateHoliday_Backward:=time_window]
  
  all_data[, Tmp_Date:=NULL]
}

add_fw_features <- function(time_window){
  # Promo
  all_data[.N:1, Tmp_Date:=cummin(ifelse(Promo!=0, Date, future_date)), by=Store]
  all_data[, Tmp_Date:=as.Date(Tmp_Date)]
  all_data[, Promo_Forward:=as.integer(Tmp_Date-Date)]
  all_data[Promo_Forward>time_window, Promo_Forward:=time_window]
  
  # SchoolHoliday
  all_data[.N:1, Tmp_Date:=cummin(ifelse(SchoolHoliday!=0, Date, future_date)), by=Store]
  all_data[, Tmp_Date:=as.Date(Tmp_Date)]
  all_data[, SchoolHoliday_Forward:=as.integer(Tmp_Date-Date)]
  all_data[SchoolHoliday_Forward>time_window, SchoolHoliday_Forward:=time_window]
  
  # StateHoliday
  all_data[.N:1, Tmp_Date:=cummin(ifelse(StateHolidayN!=0, Date, future_date)), by=Store]
  all_data[, Tmp_Date:=as.Date(Tmp_Date)]
  all_data[, StateHoliday_Forward:=as.integer(Tmp_Date-Date)]
  all_data[StateHoliday_Forward>time_window, StateHoliday_Forward:=time_window]
  
  all_data[, Tmp_Date:=NULL]
}

add_fb_features <- function(time_window=7){
  add_bw_features(time_window)
  add_fw_features(time_window)
}

add_period_count_features <- function(period=7){
  
  all_data[, SchoolHoliday_Count_BW:=rollsumr(SchoolHoliday, period, na.pad=T), by=Store]
  all_data[is.na(SchoolHoliday_Count_BW), SchoolHoliday_Count_BW:=cumsum(SchoolHoliday), by=Store]
  
  all_data[, Is_StateHoliday:=StateHolidayN!=0]
  all_data[, StateHoliday_Count_BW:=rollsumr(Is_StateHoliday, period, na.pad=T), by=Store]
  all_data[is.na(StateHoliday_Count_BW), StateHoliday_Count_BW:=cumsum(Is_StateHoliday), by=Store]
  
  all_data[, Promo_Count_BW:=rollsumr(Promo, period, na.pad=T), by=Store]
  all_data[is.na(Promo_Count_BW), Promo_Count_BW:=cumsum(Promo), by=Store]
  
  all_data[, Open_Count_BW:=rollsumr(Open, period, na.pad=T), by=Store]
  all_data[is.na(Open_Count_BW), Open_Count_BW:=cumsum(Open), by=Store]
  

  
  all_data[.N:1, SchoolHoliday_Count_FW:=rollsumr(SchoolHoliday, period, na.pad=T), by=Store]
  all_data[is.na(SchoolHoliday_Count_FW), SchoolHoliday_Count_FW:=cumsum(SchoolHoliday), by=Store]
  
  all_data[.N:1, StateHoliday_Count_FW:=rollsumr(Is_StateHoliday, period, na.pad=T), by=Store]
  all_data[is.na(StateHoliday_Count_FW), StateHoliday_Count_FW:=cumsum(Is_StateHoliday), by=Store]
  
  all_data[.N:1, Promo_Count_FW:=rollsumr(Promo, period, na.pad=T), by=Store]
  all_data[is.na(Promo_Count_FW), Promo_Count_FW:=cumsum(Promo), by=Store]
  
  all_data[.N:1, Open_Count_FW:=rollsumr(Open, period, na.pad=T), by=Store]
  all_data[is.na(Open_Count_FW), Open_Count_FW:=cumsum(Open), by=Store]

}

add_promo_decay_feature <- function(time_window=5){
  all_data[, diff_Promo:=Promo - shift(Promo, fill=0), by=Store]
  all_data[, PromoStartedDate:=cummax(ifelse(diff_Promo==1, Date, past_date)), by=Store]
  all_data[, PromoStartedDate:=as.Date(PromoStartedDate)]
  all_data[, PromoDecay:=as.integer(Date-PromoStartedDate)]
  all_data[PromoDecay>time_window, PromoDecay:=time_window]
  
  all_data[, diff_Promo:=NULL]
  all_data[, PromoStartedDate:=NULL]
  
}

view_sales <- function(store_id, start_date = NULL, end_date = NULL){
  if(is.null(start_date)){
    start_date = '1970-1-1'
  }
  if(is.null(end_date)){
    end_date = '2100-1-1'
  }
  start_date = as.Date(start_date)
  end_date = as.Date(end_date)
  dt = all_data[Store==store_id & Sales>0 & (Date>=start_date) & (Date<=end_date), .(Date, Sales)]
  
  ggplot(dt, aes(x=Date, y=Sales)) + geom_line() + geom_point()
  
  #ggplot(all_data[Store==store_id & (Date>=start_date) & (Date<=end_date) & (Sales>0), .(Date, Sales)], aes(x=Date, y=Sales)) + 
  #  geom_line()
  
}



cal_shopavg <- function(){
  dt = all_data[is.na(Id)]
  shopavg = dt[, .(Total_Sales=sum(Sales)) ,keyby='Store']
  shopavg[, Open_Days:=dt[Open==1, .N, keyby='Store']$N]
  shopavg[, Total_Customers:=dt[, sum(Customers), keyby='Store']$V1]
  shopavg[, Sales_Per_Day:=Total_Sales/Open_Days]
  shopavg[, Customers_Per_Day:=Total_Customers/Open_Days]
  shopavg[, Sales_Per_Customer:=Sales_Per_Day/Customers_Per_Day]
  
  sales_promo = dt[Promo==1, sum(Sales), keyby='Store']
  shopavg[, Sales_Promo:=sales_promo$V1/Total_Sales]
  
  sales_schoolholiday = dt[SchoolHoliday==1, sum(Sales), keyby='Store']
  shopavg[, Sales_SchoolHoliday:=sales_schoolholiday$V1/Total_Sales]
  
  sales_holiday = dt[StateHolidayN!=0, sum(Sales), keyby='Store']
  shopavg[, Sales_Holiday:=sales_holiday$V1/Total_Sales]
  
  sales_saturday = dt[DayOfWeek==6, sum(Sales), keyby='Store']
  shopavg[, Sales_Saturday:=sales_saturday$V1/Total_Sales]
  
  shopavg[, Open_Ratio:=Open_Days/942]
  schoolholiday_days = dt[SchoolHoliday==1, .N, keyby='Store']
  shopavg[, SchoolHoliday_Ratio:=schoolholiday_days$N/942]
  
  shopavg = shopavg[, lapply(.SD, function(X) (X-min(X))/(max(X) - min(X))), 
                    .SDcols=c("Sales_Per_Day", "Customers_Per_Day", "Sales_Per_Customer", "Sales_Promo", "Sales_Holiday",
                              "Sales_Saturday", "Open_Ratio", "SchoolHoliday_Ratio")]
  shopavg[, Store:=1:1115]
  
}

save_dt <- function(dt, features, filename, filter_outliers=FALSE){
  if(filter_outliers){
    train_dt = dt[is.na(Id)&Sales>0&Delete==FALSE, features, with=F]
    
  }else{
    train_dt = dt[is.na(Id)&Sales>0, features, with=F]
    
  }
  test_dt = dt[!is.na(Id), features, with=F]
  test_dt = test_dt[order(Id)]
  
  h5f = H5Fcreate(filename)
  h5write(as.matrix(train_dt, byrow=TRUE), h5f, name='train')
  h5write(as.matrix(test_dt, byrow=TRUE), h5f, name='test')
  h5write(features, h5f, name='features')
  H5close()
}



save_seq_data <- function(){
  open_data = lag_lead_features(all_data, 'Open', fill=1, time_window = 7)
  promo_data = lag_lead_features(all_data, 'Promo', fill=0, time_window = 7)
  stateholiday_data = lag_lead_features(all_data, 'StateHolidayN', fill=0, time_window = 7)
  schoolholiday_data = lag_lead_features(all_data, 'SchoolHoliday', fill=0, time_window = 7)
  
  h5file = H5Fcreate('../output/seq_data.h5')
  h5g_train = H5Gcreate(h5file, 'train')
  h5g_test = H5Gcreate(h5file, 'test')
  
  mask = is.na(all_data$Id) & (all_data$Sales>0)
  h5write(as.matrix(open_data[mask], byrow=T), h5g_train, 'open')
  h5write(as.matrix(promo_data[mask], byrow=T), h5g_train, 'promo')
  h5write(as.matrix(schoolholiday_data[mask], byrow=T), h5g_train, 'schoolholiday')
  h5write(as.matrix(stateholiday_data[mask], byrow=T), h5g_train, 'stateholiday')
  
  mask = !is.na(all_data$Id)
  h5write(as.matrix(open_data[mask], byrow=T), h5g_test, 'open')
  h5write(as.matrix(promo_data[mask], byrow=T), h5g_test, 'promo')
  h5write(as.matrix(schoolholiday_data[mask], byrow=T), h5g_test, 'schoolholiday')
  h5write(as.matrix(stateholiday_data[mask], byrow=T), h5g_test, 'stateholiday')
  H5close()
}



get_mean_sales <- function(){
  mean_sales = all_data[, .(Mean_Sales=mean(Sales), Mean_Customers=mean(Customers)), by=.(Store, Year, Month)]
  mean_sales[, Quar_Mean_Sales:=shift(rollmean(Mean_Sales, 3, na.pad=T, align='right'),2), by=.(Store)]
  mean_sales[, Quar_Mean_Customers:=shift(rollmean(Mean_Customers, 3, na.pad=T, align='right'),2), by=.(Store)]
  mean_sales[is.na(Quar_Mean_Sales),Quar_Mean_Sales:=Mean_Sales]
  mean_sales[is.na(Quar_Mean_Customers),Quar_Mean_Customers:=Mean_Customers]
  all_data2=merge(all_data, mean_sales[, .(Store, Year, Month, Quar_Mean_Sales, Quar_Mean_Customers)],
                  by=c('Store','Year', 'Month'), all=T)

  mean_sales_by_promo = all_data[, .(Mean_Sales=mean(Sales), Mean_Customers=mean(Customers)), 
                                 by=.(Store,Promo,Year, Month)]
  mean_sales_by_promo[, Quar_Mean_Sales_P:=shift(rollmean(Mean_Sales, 3, na.pad=T, align='right'),2),
                      by=.(Store, Promo)]
  mean_sales_by_promo[, Quar_Mean_Customers_P:=shift(rollmean(Mean_Customers, 3, na.pad=T, align='right'),2),
                      by=.(Store, Promo)]
  mean_sales_by_promo[is.na(Quar_Mean_Sales_P),Quar_Mean_Sales_P:=Mean_Sales,by=.(Store,Promo)]
  mean_sales_by_promo[is.na(Quar_Mean_Customers_P),Quar_Mean_Customers_P:=Mean_Customers,by=.(Store,Promo)]
  all_data2=merge(all_data2, mean_sales_by_promo[, .(Store, Promo, Year, Month, Quar_Mean_Sales_P, Quar_Mean_Customers_P)],
                                                 by=c('Store', 'Promo', 'Year', 'Month'), all=T)
  
  mean_sales_by_dayofweek = all_data[, .(Mean_Sales=mean(Sales), Mean_Customers=mean(Customers)),
                                     by=.(Store, Year, Month, DayOfWeek)]
  mean_sales_by_dayofweek[, Quar_Mean_Sales_DW:=shift(rollmean(Mean_Sales, 3, na.pad=T, align='right'),2),
                      by=.(Store, DayOfWeek)]
  mean_sales_by_dayofweek[, Quar_Mean_Customers_DW:=shift(rollmean(Mean_Customers, 3, na.pad=T, align='right'),2),
                          by=.(Store, DayOfWeek)]
  mean_sales_by_dayofweek[is.na(Quar_Mean_Sales_DW),Quar_Mean_Sales_DW:=Mean_Sales,by=.(Store,DayOfWeek)]
  mean_sales_by_dayofweek[is.na(Quar_Mean_Customers_DW),Quar_Mean_Customers_DW:=Mean_Customers,by=.(Store,DayOfWeek)]  
  all_data2=merge(all_data2, mean_sales_by_dayofweek[, .(Store, DayOfWeek, Year, Month, Quar_Mean_Sales_DW, Quar_Mean_Customers_DW)],
                  by=c('Store', 'DayOfWeek', 'Year', 'Month'), all=T)
  
  features_Quar_Mean = c("Quar_Mean_Sales", "Quar_Mean_Customers", "Quar_Mean_Sales_P", "Quar_Mean_Customers_P",
                         "Quar_Mean_Sales_DW","Quar_Mean_Customers_DW")
  
  all_data2[, c(features_Quar_Mean):=lapply(.SD, log1p), .SDcols=features_Quar_Mean]

  scaled_mean = scaled_mean[, lapply(.SD, function(X) (X-min(X))/(max(X) - min(X)))]
  
  all_data2[, `:=`(c("Quar_Mean_Sales", "Quar_Mean_Customers", "Quar_Mean_Sales_P", "Quar_Mean_Customers_P",
                     "Quar_Mean_Sales_DW","Quar_Mean_Customers_DW"),
                   scaled_mean)]
}


outliers = data.table(Store=105, Date='2013-5-8')
outliers = rbindlist(list(outliers, 
  list(163, '2013-5-8'),
  list(172, '2013-4-5'),
  list(364, '2013-5-27'),
  list(589, '2013-5-31'),
  list(633, '2014-1-18'),
  list(676, '2013-4-15'),
  list(681, '2013-5-31'),
  list(700, '2013-7-5'),
  list(708, '2013-5-8'),
  list(709, '2013-2-4'),
  list(730, '2014-1-18'),
  list(764, '2013-5-8'),
  list(837, '2014-3-18'),
  list(845, '2013-5-8'),
  list(861, '2013-5-8'),
  list(882, '2013-5-8'),
  list(969, '2013-4-15'),
  list(986, '2013-5-8'),
  list(192, '2015-1-12'),
  list(263, '2015-1-12'),
  list(500, '2015-1-12'),
  list(797, '2015-1-12'),
  list(815, '2015-1-12'),
  list(825, '2015-1-12')
  ))

outliers[, Date:=as.Date(Date)]

mask_outliers <- function(dt){
  dt[, Delete:=FALSE]
  for(i in 1:nrow(outliers)){
    store = outliers[i, Store]
    date = outliers[i, Date]
    
    dt[Store==store&Date<=date, Delete:=TRUE]
  }
}

to_date <- function(s){
  as.POSIXct(s*1e9, origin='1970-1-1')
}

add_long_closed_features <- function(dt, closed_days_threshold=3, time_window=7){
  tmp = dt[Open==1, .(Store, Date)]
  tmp2 = data.table(Store=1:1115, Date=as.Date('2012-12-31'))
  tmp3 = data.table(Store=stores_in_test, Date=as.Date('2015-9-18'))
  tmp4 = data.table(Store=stores_not_in_test, Date=as.Date('2015-8-1'))
  tmp = rbindlist(list(tmp2, tmp, tmp3, tmp4))
  setkey(tmp, 'Date', 'Store')
  
  tmp[, Last_Open_Date:=shift(Date, 1, fill=NA, type='lag'), by=Store]
  tmp[, Closed_Days:=as.integer(Date - Last_Open_Date)-1]
  tmp = tmp[Closed_Days>=closed_days_threshold&Closed_Days<185]
  dt[, Before_Long_Closed:=0]
  dt[, After_Long_Closed:=0]
  
  cat('nrow(tmp):', nrow(tmp), '\n')
  for(i in 1:nrow(tmp)){
    store = tmp[i, Store]
    last_open_date = tmp[i, Last_Open_Date]
    reopen_date = tmp[i, Date]
    
    cat('Row ', i, ': ', store, last_open_date, reopen_date, '\n')
    dt[Store==store&Date<=last_open_date&Date>last_open_date-time_window,
        Before_Long_Closed:=last_open_date-Date+1]
    dt[Store==store&Date>=reopen_date&Date<reopen_date+time_window,
        After_Long_Closed:=Date-reopen_date+1]
  }
}



# add_MA_features <- function(weeks=8){
#   all_data[, MA_Sales_W8:=rollmeanr(Sales,weeks*8,), by=Store]
# }


summer_holidays = fread('../input/GermanSummerHoliday.csv')
summer_holidays[, Start_Date:=as.Date(Start_Date)]
summer_holidays[, End_Date:=as.Date(End_Date)]
summer_holidays[, Days:=End_Date-Start_Date+1]
summer_holidays[, Year:=year(Start_Date)]
summer_holidays[State=='NI', State:='HB,NI']

add_summer_holiday_features <- function(){
  for(i in 1:nrow(summer_holidays)){
    all_data[Year==summer_holidays[i, Year] & State==summer_holidays[i, State], 
             c('Summer_Start_Date', 'Summer_End_Date'):=list(summer_holidays[i, Start_Date], summer_holidays[i, End_Date])]
  }
  all_data[, Before_SummerHoliday_Start:=as.integer(Summer_Start_Date-Date)]
  all_data[Before_SummerHoliday_Start<0, Before_SummerHoliday_Start:=0]
  all_data[Before_SummerHoliday_Start>15, Before_SummerHoliday_Start:=15]
  
  all_data[, After_SummerHoliday_Start:=as.integer(Date-Summer_Start_Date)]
  all_data[After_SummerHoliday_Start<0, After_SummerHoliday_Start:=0]
  all_data[After_SummerHoliday_Start>15, After_SummerHoliday_Start:=15]
  
  all_data[, Before_SummerHoliday_End:=as.integer(Summer_End_Date-Date)]
  all_data[Before_SummerHoliday_End<0, Before_SummerHoliday_End:=0]
  all_data[Before_SummerHoliday_End>15, Before_SummerHoliday_End:=15]
  
  all_data[, After_SummerHoliday_End:=as.integer(Date-Summer_End_Date)]
  all_data[After_SummerHoliday_End<0, After_SummerHoliday_End:=0]
  all_data[After_SummerHoliday_End>15, After_SummerHoliday_End:=15]
  
  all_data[is.na(Before_SummerHoliday_Start), Before_SummerHoliday_Start:=15]
  all_data[is.na(After_SummerHoliday_Start), After_SummerHoliday_Start:=15]
  all_data[is.na(Before_SummerHoliday_End), Before_SummerHoliday_End:=15]
  all_data[is.na(After_SummerHoliday_End), After_SummerHoliday_End:=15]
  
}

sales_delta <- function(delta=4){
  mean_log_sales = mean(log_sales)
  std_log_sales = sd(log_sales)
  min_bound = exp(mean_log_sales-delta* std_log_sales)
  max_bound = exp(mean_log_sales + delta*std_log_sales)
  all_data[(Sales>0 & Sales<min_bound) | Sales>max_bound, .N]
}

