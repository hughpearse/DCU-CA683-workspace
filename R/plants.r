iris_md = read.csv("/home/ciara/Sandbox/R_code/iris_with_missing_data.csv")
attach(iris_md)
summary(iris_md)


###### This looks useful as it shows 
#Found online at https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/matplot.html
nam.var <- colnames(iris_md)[-5]
nam.spec <- as.character(iris_md[1+50*0:2, "Species"])
iris_md.S <- array(NA, dim = c(50,4,3),
                dimnames = list(NULL, nam.var, nam.spec))
for(i in 1:3) iris_md.S[,,i] <- data.matrix(iris_md[1:50+50*(i-1), -5])

matplot(iris_md.S[, "Petal.Length",], iris_md.S[, "Petal.Width",], pch = "SCV",
        col = rainbow(3, start = 0.8, end = 0.1),
        sub = paste(c("S", "C", "V"), dimnames(iris_md.S)[[3]],
                    sep = "=", collapse= ",  "),
        main = "Fisher's Iris Data")

####### Regression analysis ######

#Doesn't seem to be regression for petal width, petal length
plot(Petal.Width[Species=="setosa"]~Petal.Length[Species=="setosa"])
abline(coefficients(lm(Petal.Width[Species=="setosa"]~Petal.Length[Species=="setosa"])))
lm(Petal.Width[Species=="setosa"]~Petal.Length[Species=="setosa"]) # 0.18114
plot(Petal.Width[Species=="virginica"]~Petal.Length[Species=="virginica"])
abline(coefficients(lm(Petal.Width[Species=="virginica"]~Petal.Length[Species=="virginica"])))
lm(Petal.Width[Species=="virginica"]~Petal.Length[Species=="virginica"]) # 0.1555
plot(Petal.Width[Species=="versicolor"]~Petal.Length[Species=="versicolor"])
abline(coefficients(lm(Petal.Width[Species=="versicolor"]~Petal.Length[Species=="versicolor"])))
lm(Petal.Width[Species=="versicolor"]~Petal.Length[Species=="versicolor"]) # 0.33

#Regression present
plot(Sepal.Width[Species=="setosa"]~Sepal.Length[Species=="setosa"])
abline(coefficients(lm(Sepal.Width[Species=="setosa"]~Sepal.Length[Species=="setosa"])))
lm(Sepal.Width[Species=="setosa"]~Sepal.Length[Species=="setosa"]) # 0.7985
#Does not appear to be regression
plot(Sepal.Width[Species=="virginica"]~Sepal.Length[Species=="virginica"]) 
abline(coefficients(lm(Sepal.Width[Species=="virginica"]~Sepal.Length[Species=="virginica"])))
lm(Sepal.Width[Species=="virginica"]~Sepal.Length[Species=="virginica"]) # 0.2471
plot(Sepal.Width[Species=="versicolor"]~Sepal.Length[Species=="versicolor"])
abline(coefficients(lm(Sepal.Width[Species=="versicolor"]~Sepal.Length[Species=="versicolor"])))
lm(Sepal.Width[Species=="versicolor"]~Sepal.Length[Species=="versicolor"]) # 0.3154

