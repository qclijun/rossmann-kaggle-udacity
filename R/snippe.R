trend_states = unique(trend_data$State)

for(s in trend_states){
  val = (trend_data[State==s & Year==2013 & WeekOfYear==52, Trend_Val] + trend_data[State==s & Year==2014 & WeekOfYear==1, Trend_Val])/2
  val2 = (trend_data[State==s & Year==2014 & WeekOfYear==52, Trend_Val] + trend_data[State==s & Year==2015 & WeekOfYear==1, Trend_Val])/2
  row1 = list(s, 2013, 53, val)
  row2 = list(s, 2014, 53, val2)
  trend_data <- rbindlist(list(trend_data, row1, row2))
}

for(s in trend_states[-1]){
  val = trend_data[State==s&Year==2013&WeekOfYear==53, Trend_Val]
  all_data[State==s&Year==2013&WeekOfYear==53, Trend_Val_State:=val]
  
  
  val = trend_data[State==s&Year==2014&WeekOfYear==53, Trend_Val]
  all_data[State==s&Year==2014&WeekOfYear==53, Trend_Val_State:=val]
  
}

val = trend_data[State=='DE'&Year==2013&WeekOfYear==53, Trend_Val]
all_data[Year==2013&WeekOfYear==53, Trend_Val_DE:=val]
val = trend_data[State=='DE'&Year==2014&WeekOfYear==53, Trend_Val]
all_data[Year==2014&WeekOfYear==53, Trend_Val_DE:=val]



dt2 = all_data[Open==1, .(Store, Date)]
dt2[, Last_Open_Date:=shift(Date, 1, fill=NA, type='lag'), by=Store]
dt2[, diff_Date:=as.integer(Date-Last_Open_Date), by=Store]
long_closed_days = dt2[, unique(diff_Date)]
dt2[, Long_Closed:=diff_Date>=4]
