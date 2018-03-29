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



ggplot(mean_sales[Store==4|Store==134|Store==1011|Store==980|Store==580], aes(x=Year_Month, y=Mean_Sales, group=Store)) + 
  geom_line(aes(color=factor(Store))) + 
  geom_point(aes(color=factor(Store),shape=factor(Store)))

##################################################################################
# Model Visualization

# plot predicted sales
sales_pred = fread('../sales_pred.csv')
sales_pred[, Date:=as.Date(Date)]
# add zero sales
sales_pred = merge(sales_pred[, !c('Sales')], all_data[,.(Date, Store, Open, Sales)], 
                    all.y=T, all.x=F, by=c('Date', 'Store'))
sales_pred[Sales==0, Sales_Pred:=0]
sales_pred[Open==0, Sales_Pred:=0]
ggplot(sales_pred[Store==174&Date>as.Date('2015-1-1')]) + 
  geom_point(aes(Date, Sales), color='red') + 
  geom_line(aes(Date, Sales_Pred)) + 
  geom_point(aes(Date, Sales_Pred2), color='blue', shape=2)

sales_pred100 = fread('../sales_pred_100%.csv')
sales_pred100[, Date:=as.Date(Date)]
# add zero sales
sales_pred100 = merge(sales_pred100[, !c('Sales')], all_data[,.(Date, Store, Open, Sales)], 
                   all.y=T, all.x=F, by=c('Date', 'Store'))
sales_pred100[Sales==0, Sales_Pred:=0]
sales_pred100[Open==0, Sales_Pred:=0]

sales_pred[, Sales_Pred100:=sales_pred100$Sales_Pred]
ggplot(sales_pred[Store==78&Date>as.Date('2015-1-1')]) + 
  geom_point(aes(Date, Sales), color='red') + 
  geom_line(aes(Date, Sales_Pred)) + 
  geom_line(aes(Date, Sales_Pred100), color='blue', name='Model 2')


# plot embedding features
embed_feas = h5read('../embedding_features.h5', '/train_embed')
embed_feas = t(embed_feas)


importance = fread('../importance.csv')
importance_dt = importance[order(-importance)]

importance_dt[, name:=factor(importance_dt$name, levels=importance_dt$name)]

ggplot(importance_dt[1:50], aes(name, importance)) + 
  geom_bar(stat='identity') + coord_flip() 

importance_split = fread('../importance_split.csv')
importance_split_dt = importance_split[order(-importance)]

importance_split_dt[, name:=factor(importance_split_dt$name, levels=importance_split_dt$name)]

ggplot(importance_split_dt[1:50], aes(name, importance)) + 
  geom_bar(stat='identity') + coord_flip() 
    