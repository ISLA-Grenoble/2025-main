# load MASS to to use mvrnorm for sampling multivariate normal
library("MASS")
# create a function for sampling from p(x,y)
generate_dataset <- function(nsamples, eps, rho_0, rho_1)
{
  # set the seed at each call for reproducibility
  set.seed(42)
  # parameters for class 0
  mu_0 = c(0, 0)
  cv_0 = rbind(c(1, rho_0*2), c(rho_0*2, 4))
  # parameters for class 1
  mu_1 = c(1, 1)/sqrt(2) * eps
  cv_1 = rbind(c(1, rho_1*2), c(rho_1*2, 4))
  # sample data points -- note the effect of the seed being fixed
  X_0 = mvrnorm(n=nsamples, mu=mu_0, Sigma=cv_0)
  X_1 = mvrnorm(n=nsamples, mu=mu_1, Sigma=cv_1)
  X = rbind(X_0, X_1)
  y = c(rep(0, nsamples), rep(1, nsamples))
  df = data.frame(cbind(X, y))
  colnames(df) = c('X1', 'X2', 'y')
  return(df)
}
# create a function to get score for a given eps
get_error_lrg_eps <- function(eps)
{
  # generate a dataset for training
  df_train = generate_dataset(nsamples=100, eps=eps, rho_0=0.5, rho_1=0.5)
  # generate a dataset for testing
  df_test = generate_dataset(nsamples=2500, eps=eps, rho_0=0.5, rho_1=0.5)
  # train the logistic regression classifier
  logreg = glm(y ~ ., data=df_train, family=binomial)
  # get the prob predictions of the classifier on the test data
  y_pred_prob = predict(logreg, newdata=df_test, type="response")
  # cast the prob predictions into class 0 or 1
  y_pred = as.integer(y_pred_prob > 0.5)
  # get the true labels of the test dataset
  y_test = df_test[,3]
  # calculate the average error rate
  error_lrg = mean(y_test != y_pred)
  return(error_lrg)
}
# loop over different values for eps
eps_list = seq(0, 4, length.out=50)
error_lrg_list = c()
for (eps in eps_list)
{

  # get the errors over multiple realizations to smooth things out
  nrzt = 10
  error_lrg_avg = 0.0
  for (rzt in 1:10) 
    {error_lrg_avg = error_lrg_avg + get_error_lrg_eps(eps)}
  error_lrg_avg = error_lrg_avg / nrzt
  
  # append result
  error_lrg_list = c(error_lrg_list, error_lrg_avg)
  
  # only generate a plot every 10 values of eps
  if(length(error_lrg_list) %% 10 == 0)
  {

    # instantiate a pdf file to save the plot
    filename = sprintf("fig-CM5-logreg-eps_%0.1f.pdf", eps)    
    pdf(filename, width=12.0, height=6.8)

    # prepare a subplot
    par(mfrow = c(1, 2))
    
    # plot a scatter plot on the left column with data freshly generated
    df_train = generate_dataset(
      nsamples=250, 
      eps=eps, 
      rho_0=0.5, 
      rho_1=0.5)
    y = df_train[,3]
    plot(df_train[,1:2], 
         col='white',
         xlim=c(-5, +5),
         ylim=c(-8, +8))
    points(df_train[y==0,1:2],
           col='#1f77b4',
           pch=16,
           cex=1.5)
    points(df_train[y==1,1:2],
           col='#ff7f0e', 
           pch=16,
           cex=1.5)
    title(sprintf("Samples from p(x,y) with eps = %0.1f", eps))      
    
    # plot the error curve as function of eps on the right column
    plot(eps_list,
         0.5*rep(1, length(eps_list)),
         col='white',
         type='l',
         xlab='eps',
         ylab='error',
         xlim=c(0, 4),
         ylim=c(0, 0.5)
    )
    lines(eps_list[1:length(error_lrg_list)],
          error_lrg_list,
          col='black',
          lwd=6.0
    )
    points(eps, error_lrg_avg, col='black', pch=16, cex=1.5)
    abline(v=eps, col='black', lty=2)
    title('Error curve for logistic regression')  
    
    dev.off()
  }
}

# # true conditional probability P(Y = 1 | X = x) -- this is what we are aiming for
# prob_true <- matrix(ESL.mixture$prob, nrow=length(x1_array), ncol=length(x2_array))
# 
# # estimate the kNN classifier for this data
# prob_knn_15 <- knn(train=df_train[,c(2,3)], test=df_test, cl=df_train[,1], k=15, prob=TRUE)
# # getting the probability predictions for each point in the feature space
# prob_knn_15 <- matrix(prob_knn_15, nrow=length(x1_array), ncol=length(x2_array))
# 
# # estimating the logistic regression classifier for this data
# logreg <- glm(y_train ~ ., data=df_train, family=binomial)
# y_pred <- predict.glm(logreg, newdata=df_test, type="response")
# # getting the probability predictions for each point in the feature space
# prob_logreg <- matrix(y_pred, nrow=length(x1_array), ncol=length(x2_array))