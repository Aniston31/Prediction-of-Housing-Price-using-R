# ============================================================
# PROJECT: PREDICTION OF HOUSING PRICE USING R
# Methods: Multiple Linear Regression, Log Transformation,
#          and Box-Cox Transformation
# Dataset: Housing.csv
# ============================================================

# 1. Install packages (run only once if not already installed)
# install.packages("MASS")
# install.packages("car")

# 2. Load required packages
library(MASS)  # Box-Cox transformation
library(car)   # VIF / multicollinearity test

# 3. Import the dataset
# Keep Housing.csv in the same folder as this R script.
data <- read.csv("Housing.csv", stringsAsFactors = FALSE)

# Check the dataset
head(data)
str(data)
summary(data)
dim(data)       # Expected: 545 observations and 13 variables

# 4. Prepare the data
# Convert categorical columns into factors.
categorical_cols <- c(
  "mainroad", "guestroom", "basement", "hotwaterheating",
  "airconditioning", "prefarea", "furnishingstatus"
)

data[categorical_cols] <- lapply(data[categorical_cols], factor)

# Check for missing values
colSums(is.na(data))

# Remove rows containing missing values, if any
data <- na.omit(data)

# Check that house prices are positive before transformations
if (any(data$price <= 0)) {
  stop("The price column contains zero or negative values; transformations require positive prices.")
}

# 5. Test for multicollinearity
# Fit an initial model for the VIF test
initial_model <- lm(price ~ ., data = data)

cat("\\nVIF / GVIF results:\\n")
print(car::vif(initial_model))

# For factor variables, car::vif() may report GVIF.
# A GVIF^(1/(2*Df)) value close to 1 generally indicates
# low multicollinearity. Investigate unusually high values.

# ============================================================
# MODEL 1: MULTIPLE LINEAR REGRESSION
# ============================================================

model1 <- lm(price ~ ., data = data)

cat("\\nMODEL 1: MULTIPLE LINEAR REGRESSION\\n")
print(summary(model1))

# Four standard regression diagnostic plots
par(mfrow = c(2, 2))
plot(model1)
par(mfrow = c(1, 1))

# ============================================================
# MODEL 2: LOG TRANSFORMATION
# The project report applies log() to the bedrooms predictor.
# This keeps price as the dependent variable.
# ============================================================

# Avoid log(0), if any bedroom values are zero
if (any(data$bedrooms <= 0)) {
  stop("The bedrooms column contains zero or negative values; log(bedrooms) cannot be calculated.")
}

log_model <- lm(
  price ~ area + log(bedrooms) + bathrooms + stories +
    mainroad + guestroom + basement + hotwaterheating +
    airconditioning + parking + prefarea + furnishingstatus,
  data = data
)

cat("\\nMODEL 2: LOG TRANSFORMATION OF BEDROOMS\\n")
print(summary(log_model))

par(mfrow = c(2, 2))
plot(log_model)
par(mfrow = c(1, 1))

# ============================================================
# MODEL 3: BOX-COX TRANSFORMATION OF HOUSE PRICE
# ============================================================

# Find the optimal lambda using the original regression model
bc_result <- MASS::boxcox(
  model1,
  lambda = seq(-2, 2, by = 0.1),
  main = "Box-Cox Transformation"
)

# Lambda that maximises the log-likelihood
lambda_opt <- bc_result$x[which.max(bc_result$y)]
cat("\\nOptimal Box-Cox lambda:", lambda_opt, "\\n")

# Transform the dependent variable
if (abs(lambda_opt) < 0.05) {
  data$price_bc <- log(data$price)
} else {
  data$price_bc <- (data$price^lambda_opt - 1) / lambda_opt
}

# Fit the model using the transformed price
model3 <- lm(
  price_bc ~ area + bedrooms + bathrooms + stories +
    mainroad + guestroom + basement + hotwaterheating +
    airconditioning + parking + prefarea + furnishingstatus,
  data = data
)

cat("\\nMODEL 3: BOX-COX TRANSFORMED MODEL\\n")
print(summary(model3))

par(mfrow = c(2, 2))
plot(model3)
par(mfrow = c(1, 1))

# ============================================================
# MODEL COMPARISON
# ============================================================

# R-squared and adjusted R-squared for each model
model_comparison <- data.frame(
  Model = c("Initial MLR", "Log bedrooms", "Box-Cox price"),
  R_squared = c(
    summary(model1)$r.squared,
    summary(log_model)$r.squared,
    summary(model3)$r.squared
  ),
  Adjusted_R_squared = c(
    summary(model1)$adj.r.squared,
    summary(log_model)$adj.r.squared,
    summary(model3)$adj.r.squared
  ),
  Residual_Standard_Error = c(
    sigma(model1),
    sigma(log_model),
    sigma(model3)
  )
)

cat("\\nMODEL COMPARISON\\n")
print(model_comparison)

# Note:
# Do not directly compare AIC values from models whose dependent
# variables are on different transformed scales (price versus
# Box-Cox(price)) without accounting for the transformation's
# likelihood/Jacobian. R-squared values also refer to different
# response scales, so interpret them with care.

# ============================================================
# PREDICT HOUSE PRICE FOR A NEW HOUSE
# Enter values that match the dataset's variable names and
# factor levels. Example values below are illustrative only.
# ============================================================

# new_house <- data.frame(
#   area = 5000,
#   bedrooms = 3,
#   bathrooms = 2,
#   stories = 2,
#   mainroad = factor("yes", levels = levels(data$mainroad)),
#   guestroom = factor("no", levels = levels(data$guestroom)),
#   basement = factor("no", levels = levels(data$basement)),
#   hotwaterheating = factor("no", levels = levels(data$hotwaterheating)),
#   airconditioning = factor("yes", levels = levels(data$airconditioning)),
#   parking = 1,
#   prefarea = factor("yes", levels = levels(data$prefarea)),
#   furnishingstatus = factor("furnished",
#                             levels = levels(data$furnishingstatus))
# )
#
# predicted_price <- predict(model1, newdata = new_house)
# cat("Predicted house price:", predicted_price, "\\n")

# End of project
