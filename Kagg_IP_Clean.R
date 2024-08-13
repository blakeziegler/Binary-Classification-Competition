# Load necessary libraries
library(tidyverse)
library(dplyr)
library(data.table)

# Summarize the datasets
summary(train)
summary(test)

# Rename columns in the training dataset
train_clean <- train %>%
  rename(
    dl = Driving_License,
    r_code = Region_Code,
    prev_I = Previously_Insured,
    v_age = Vehicle_Age,
    v_dam = Vehicle_Damage,
    a_prem = Annual_Premium,
    channel = Policy_Sales_Channel
  )

# Rename columns in the test dataset
test_clean <- test %>%
  rename(
    dl = Driving_License,
    r_code = Region_Code,
    prev_I = Previously_Insured,
    v_age = Vehicle_Age,
    v_dam = Vehicle_Damage,
    a_prem = Annual_Premium,
    channel = Policy_Sales_Channel
  )

# Split the Gender column in the training dataset
train_clean <- train_clean %>%
  mutate(
    Male = ifelse(Gender == "Male", 1, 0),
    Female = ifelse(Gender == "Female", 1, 0)
  ) %>%
  select(-Gender)  # Remove the original Gender column

# Split the Gender column in the test dataset
test_clean <- test_clean %>%
  mutate(
    Male = ifelse(Gender == "Male", 1, 0),
    Female = ifelse(Gender == "Female", 1, 0)
  ) %>%
  select(-Gender)  # Remove the original Gender column

train_clean <- train %>%
  select(id, r_code, channel) %>%
  right_join(train_clean, by = "id")

# Merge original r_code and channel columns back into test_clean
test_clean <- test %>%
  select(id, r_code, channel) %>%
  right_join(test_clean, by = "id")

summary(train_clean)
summary(test_clean)

train_dummies <- model.matrix(~ r_code + channel - 1, data = train_clean) %>% as.data.frame()
train_clean <- cbind(train_clean, train_dummies)

test_dummies <- model.matrix(~ r_code + channel - 1, data = test_clean) %>% as.data.frame()
test_clean <- cbind(test_clean, test_dummies)

train_clean <- train_clean[, !duplicated(colnames(train_clean))]
test_clean <- test_clean[, !duplicated(colnames(test_clean))]

summary(train_clean)


# Normalize function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Normalize specified columns in train_clean
train_clean <- train_clean %>%
  mutate(
    a_prem = normalize(a_prem),
    Vintage = normalize(Vintage),
    Age = normalize(Age)
  )

# Normalize specified columns in test_clean
test_clean <- test_clean %>%
  mutate(
    a_prem = normalize(a_prem),
    Vintage = normalize(Vintage),
    Age = normalize(Age)
  )

# Export the datasets to CSV
write.csv(train_clean, "train_clean.csv", row.names = FALSE)
write.csv(test_clean, "test_clean.csv", row.names = FALSE)

summary(train_clean)
summary(test_clean)