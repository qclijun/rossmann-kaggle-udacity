source('library.R')


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
  
  trend_data[, .(State, Year, WeekOfYear, Trend_Val)]
}

generate_forward_backward_information_accumulated <- function(data, window_size=14, only_zero=TRUE){
  columns = c('Promo', 'StateHoliday', 'SchoolHoliday')
  
}


make_stairs1 <- function(data, sep, values){
  n = length(data)
  n_sep = length(sep)
  j=1
  result = rep(values[1], n)
  for(i in 1:n_sep){
    beg=j
    while(j<=n && data[j]<sep[i]){ 
      j=j+1
    }
    result[beg:(j-1)]=values[i]
  }
  if(j<=n){
    result[j:n]=values[i+1]
  }
  result
}

make_stairs2 <- function(data, sep, values){
  n = length(data)
  n_sep = length(sep)
  j=1
  result = rep(values[1], n)
  for(i in 1:n_sep){
    beg=j
    
    while(j<=n && data[j]<=sep[i]){ 
      j=j+1
    }
    result[beg:(j-1)]=values[i]
  }
  if(j<=n){
    result[j:n]=values[i+1]
  }
  result
}

make_stairs <- function(data, sep, values, change_on_equals=TRUE){
  if(change_on_equals){
    make_stairs1(data, sep, values)
  }else{
    make_stairs2(data, sep, values)
  }
}

cal_forward <- function(dt, store_id, colname,  time_window=7){
  d = dt[.(store_id), c('Date', colname), with=F]
  dates = d$Date
  sep_dates = d[eval(as.name(colname))!=0, Date]
  stair_values = c(sep_dates, as.Date('2100-1-1'))
  x=make_stairs(dates, sep_dates, stair_values)
  x=x-dates
  x=ifelse(x>time_window+1, time_window+1, x)
  x
}

cal_backward <- function(dt, store_id, colname, time_window=7){
  d = dt[.(store_id), c('Date',colname), with=F]
  dates = d$Date
  sep_dates = d[eval(as.name(colname))!=0, Date]
  stair_values = c(as.Date('1970-1-1'), sep_dates)
  x=make_stairs(dates, sep_dates, stair_values, change_on_equals = F)
  x=dates - x
  x=ifelse(x>time_window+1, time_window+1, x)
  x
}

add_forward_backward_features <- function(dt, colnames, time_window=7){
  for(store_id in 1:1115){
    for(colname in colnames){
      nm = ifelse(colname=='StateHoliday', 'StateHolidayN', colname)
      x = cal_forward(dt, store_id, nm, time_window)
      dt[.(store_id), paste0(colname, '_', 'Forward'):=x]
      x = cal_backward(dt, store_id, nm, time_window)
      dt[.(store_id), paste0(colname, '_', 'Backward'):=x]
    }
  }
}

add_stateholiday_count_fb <- function(dt, time_window=7){
  dt[, StateHoliday_Count_Forward:=0]
  dt[, StateHoliday_Count_Backward:=0]
  has_holiday = dt[StateHolidayN!=0, .(Store, Date)]
  has_holiday_stores = has_holiday$Store
  has_holiday_dates = has_holiday$Date
  for(i in 1:length(has_holiday_dates)){
    d = has_holiday_dates[i]
    store_id = has_holiday_stores[i]
    dt[Store==store_id & Date>d & Date<=(d+7), StateHoliday_Count_Backward := StateHoliday_Count_Backward+1]
    dt[Store==store_id & Date<d & Date>=(d-7), StateHoliday_Count_Forward := StateHoliday_Count_Forward+1]
  }
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
  dt = all_data[Store==store_id & Sales>0 & (Date>=start_date) & (Date<=end_date), .(Date, Sales, Delete)]
  ggplot(dt, aes(x=Date, y=Sales, color=Delete)) + geom_line()
  
  #ggplot(all_data[Store==store_id & (Date>=start_date) & (Date<=end_date) & (Sales>0), .(Date, Sales)], aes(x=Date, y=Sales)) + 
  #  geom_line()
  
}

cal_avg_sales <- function(dt){
  total_sales = dt[is.na(Id), .(Total_Sales=sum(Sales)) ,by='Store']
  total_sales[, Open_Days:=dt[is.na(Id) & Open==1, .N, by='Store']$N]
  total_sales[, Total_Customers:=dt[is.na(Id), sum(Customers), by='Store']$V1]
  total_sales[, Sales_Per_Day:=Total_Sales/Open_Days]
  total_sales[, Customers_Per_Day:=Total_Customers/Open_Days]
  total_sales[, Sales_Per_Customer:=Sales_Per_Day/Customers_Per_Day]
  
  mi = min(total_sales$Sales_Per_Day)
  ma = max(total_sales$Sales_Per_Day)
  total_sales[, Sales_Per_Day:=(Sales_Per_Day-mi)/(ma-mi)]
  
  mi = min(total_sales$Customers_Per_Day)
  ma = max(total_sales$Customers_Per_Day)
  total_sales[, Customers_Per_Day:=(Customers_Per_Day-mi)/(ma-mi)]
  
  mi = min(total_sales$Sales_Per_Customer)
  ma = max(total_sales$Sales_Per_Customer)
  total_sales[, Sales_Per_Customer:=(Sales_Per_Customer-mi)/(ma-mi)]
  total_sales
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

lag_lead_features <- function(dt, feature_name, fill=NA, time_window=7){
  result_dt = dt[, c('Store', 'Date', feature_name), with=F]
  #setkey(result_dt, 'Store', 'Date')
  
  lag_cols = paste0(feature_name, '_Lag_', time_window:1)
  result_dt[, (lag_cols):=shift(.SD, time_window:1, type='lag', fill=fill), by='Store', .SDcols=(feature_name) ]
  
  lead_cols = paste0(feature_name, '_Lead_', 1:time_window)
  result_dt[, (lead_cols):=shift(.SD, 1:time_window, type='lead', fill=fill), by='Store', .SDcols=(feature_name)]
  
  result_dt[, c(lag_cols, feature_name, lead_cols), with=F]
  
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

fill_record <- function(dt){
  
  
  
}


add_promo_decay_feature <- function(dt){
  dt[, diff_Promo:=Promo - shift(Promo, fill=0), by='Store']
  past_date = as.Date('1970-1-1')
  future_date = as.Date('2100-1-1')
  dt[, PromoStartedDate:=cummax(ifelse(diff_Promo==1, Date, past_date))]
  dt[, PromoStartedDate:=as.Date(PromoStartedDate)]
  dt[, PromoDecay:=Date-PromoStartedDate]
  dt[PromoDecay>4, PromoDecay:=5]
  dt[, PromoDecay:=as.integer(PromoDecay)]
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


