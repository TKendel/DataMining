 
data0_original = read.csv("/Users/kenialopez/Documents/P5/DM/pivot_df.csv", header = TRUE)
data0 <- as.data.frame(data0_original)

install.packages("magrittr") 
install.packages("dplyr")    
library(magrittr) 
library(dplyr)



#GROUP by id and fill in with the MEAN per id
data <- subset(data0, select = -c(time, TIME, DATE, HOUR))
data0_filled <- data %>%
  group_by(id) %>%
  mutate(across(where(is.numeric), ~mean(., na.rm = TRUE))) %>%
  ungroup() %>%
  mutate(mood = round(mood),
    circumplex.arousal = round(circumplex.arousal),
    circumplex.valence = round(circumplex.valence))

colSums(is.na(data0_filled))


data0[names(data0_filled)] <- data0_filled

write.csv(data0, "/Users/kenialopez/Documents/P5/DM/dataset_no_NA_new.csv")






