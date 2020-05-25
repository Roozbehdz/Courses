rm(list = ls(all.names = TRUE))
library(faraway)

d = read.csv('C:/Users/10/Documents/R Scripts/Zerehsaz Data Science Course/Seatdata.csv')

mf = lm(SeatX ~ Stature + Sitting.Height + SHS + BMI + Weight + L11 + H17, data =
          d)

mr = lm(SeatX ~ 1, data = d)

anova(mr, mf)

mr = lm(SeatX ~ Sitting.Height + SHS + BMI + Weight + L11 + H17, data =
          d)

anova(mr, mf)

mr = lm(SeatX ~ Sitting.Height + BMI + Weight + L11 + H17, data = d)

anova(mr, mf)

mr = lm(SeatX ~ I(Stature + Sitting.Height) + SHS + BMI + Weight + L11 +
          H17,
        data = d)

anova(mr, mf)

t = (summary(mf)$coef[2, 1] - 0.6) / summary(mf)$coef[2, 2]

2 * (1 - pt(t, 399 - 8))

mr = lm(SeatX ~ offset(0.6 * Stature) + Sitting.Height + SHS + BMI + Weight +
          L11 + H17,
        data = d)

anova(mr, mf)

#Permutation Test
y = d[, 8]
X = d[, 1:7]

f = NA
for (i in 1:1000) {
  p = sample(length(y), length(y), replace = FALSE)
  mforg = lm(y ~ as.matrix(X))
  yperm = y[p]
  mf = lm(yperm ~ as.matrix(X))
  mr = lm(yperm ~ 1)
  h = anova(mr, mf)
  f[i] = h$F[2]
  
}


# Fit the model first
mf <- lm(SeatX ~ Stature + SHS + BMI + L11 + H17, data = d)

# Plot fitted values versus residuals
plot(mf$fitted.values,
     mf$residuals,
     xlab = "Fitted Values",
     ylab = "Residuals")
abline(h = mean(mf$residuals), col = "red")
abline(v = mean(mf$fitted.values), col = "red")

# Plot fitted values versus the absolute value of residuals
plot(mf$fitted.values,
     abs(mf$residuals),
     xlab = "Fitted Values",
     ylab = "|Residuals|")

# Check for significance
summary(lm(abs(mf$residual) ~ mf$fitted))

library(nortest)
ad.test(mf$residuals)

plot(mf$residual[1:399], xlab = "i", ylab = "e(i)")
plot(mf$residual[1:398], mf$residuals[2:399], xlab = "i", ylab = "e(i)")
