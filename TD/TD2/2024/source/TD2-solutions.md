### Exercise 1

(a)
$$
\begin{array}{rcl}
\displaystyle\dfrac{1}{N}\sum_{i = 1}^N \|\mathbf{x}_i - \bar{\mathbf{x}}\|^2 &=& \displaystyle\dfrac{1}{N}\sum_{i = 1}^N (\mathbf{x}_i - \bar{\mathbf{x}})^\top (\mathbf{x}_i - \bar{\mathbf{x}}) \\[1em]
&=& \displaystyle\dfrac{1}{N}\sum_{i = 1}^N \text{tr}\Big((\mathbf{x}_i - \bar{\mathbf{x}})^\top (\mathbf{x}_i - \bar{\mathbf{x}})\Big) \\[1em]
&=& \displaystyle\dfrac{1}{N}\sum_{i = 1}^N \text{tr}\Big((\mathbf{x}_i - \bar{\mathbf{x}}) (\mathbf{x}_i - \bar{\mathbf{x}})^\top\Big) \\[1em]
&=& \text{tr}\left(\displaystyle\dfrac{1}{N}\sum_{i = 1}^N (\mathbf{x}_i - \bar{\mathbf{x}}) (\mathbf{x}_i - \bar{\mathbf{x}})^\top\right) \\[1em]
&=& \text{tr}(\mathbf{\Sigma})
\end{array}
$$
(b)
$$
\dfrac{1}{N}\sum_{i = 1}^N \|\mathbf{x}_i\|^2 = \dfrac{1}{N}\sum_{i = 1}^N \|\mathbf{x}_i - \bar{\mathbf{x}}\|^2 = \text{tr}(\mathbf{\Sigma}) = \sum_{k = 1}^p \text{Var}(X_k) = \sum_{k = 1}^p 1 = p
$$
(c) Using Lagrangian multipliers, we have that the augmented loss function is
$$
\tilde{\mathcal{L}}(\mathbf{W}, \lambda) = \text{tr}(\mathbf{W}^\top \mathbf{\Sigma}\mathbf{W}) + \Lambda~(\mathbf{I}_q - \mathbf{W}^\top \mathbf{W})
$$
taking the partial derivative with respect to $\mathbf{W}$ we have that
$$
\dfrac{\partial \tilde{\mathcal{L}}(\mathbf{W}, \lambda)}{\partial \mathbf{W}} = 2\mathbf{\Sigma}\mathbf{W} - 2\Lambda \mathbf{W} = 0 \iff \mathbf{\Sigma}\mathbf{W}^\star = \Lambda \mathbf{}\mathbf{W}^\star
$$
So $\mathbf{W}^\star \in \mathbb{R}^{p \times q}$ is a matrix containing $q$ eigenvectors of $\mathbf{\Sigma}$, but which ones?

Note that if we plug back matrix $\mathbf{W}^\star$ into the loss function, we get
$$
\mathcal{L}(\mathbf{W}^\star) = \sum_{i = 1}^q \lambda_i
$$
To minimize $\mathcal{L}$ we should choose the eigenvectors associated to the $q$-smallest eigenvalues of $\mathbf{\Sigma}$.

---

### Exercise 2

(a) In multiple linear regression we have the model
$$
Y = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p + \varepsilon
$$
Note that if we take the expectation from both sites, we get
$$
\mathbb{E}[Y] = \beta_0 + \beta_1 \mathbb{E}[X_1] + \dots + \beta_p \mathbb{E}[X_p] + \mathbb{E}[\varepsilon]
$$
Since the predictors and observations have zero-mean, then $\beta_0 = 0$.

(b) The SVD of the data matrix is $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$ so if we plug this into the expression for $\hat{\beta}$ we get
$$
\hat{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y} = (\mathbf{V}\mathbf{D}\mathbf{U}^{\top}\mathbf{U}\mathbf{D}\mathbf{V}^\top)^{-1}\mathbf{V}\mathbf{D}\mathbf{U}^\top\mathbf{y} = (\mathbf{V}\mathbf{D}^2\mathbf{V}^\top)^{-1}\mathbf{V}\mathbf{D}\mathbf{U}^\top\mathbf{y} = \mathbf{V}\mathbf{D}^{-2}\mathbf{V}^\top\mathbf{V}\mathbf{D}\mathbf{U}^\top\mathbf{y} 
$$
so in the end we get $\hat{\beta} = \mathbf{V}\mathbf{D}^{-1}\mathbf{U}^\top \mathbf{y}$ and $\hat{\beta}_i = \displaystyle\sum_{k = 1}^p\dfrac{\mathbf{u}_i^\top \mathbf{y}}{d_k} \mathbf{v}_{ik}$

Note also that the predictions with the model are $\hat{\mathbf{y}} = \mathbf{X}\hat{\beta} = \mathbf{U}\mathbf{D}\mathbf{V}^\top\mathbf{V}\mathbf{D}^{-1}\mathbf{U}^\top \mathbf{y} = \mathbf{U}\mathbf{U}^\top\mathbf{y}$ 

(c) Matrix $\mathbf{Z}$ is the projection of the data matrix on its $q$-top principal components. We have, therefore:
$$
\mathbf{Z} = \mathbf{X}\mathbf{V}_q = \mathbf{U}\mathbf{D}\mathbf{V}^\top \mathbf{V}_q = \mathbf{U}\mathbf{D}
\left[\begin{array}{cc}
\mathbf{I}_q & \\
& \mathbf{0}_{p - q}
\end{array}\right] = \mathbf{U}\mathbf{D}_q \quad \text{where} \quad \mathbf{D}_q = \left[\begin{array}{cccc}
d_1 & & & \\
 & \ddots & & \\
& & d_q & \\
 & & & \mathbf{0}_{p-q}
\end{array}\right]
$$
We calculate the coefficients for the new regression model
$$
\hat{\gamma} = (\mathbf{Z}^\top\mathbf{Z})^{-1}\mathbf{Z}^\top \mathbf{y} = (\mathbf{D}_q\mathbf{U}^\top \mathbf{U}\mathbf{D}_q)^{-1}\mathbf{D}_q\mathbf{U}^\top\mathbf{y} = \mathbf{D}_q^{-1}\mathbf{U}^\top\mathbf{y}
$$
and if we take it back to the original space, we get
$$
\hat{\beta}^{\text{PCR}} = \mathbf{V}_q \hat{\gamma} = \mathbf{V}_q \mathbf{D}_q^{-1} \mathbf{U}^\top \mathbf{y} \quad \text{and} \quad \hat{\beta}^{\text{PCR}}_i = \sum_{k = 1}^q \dfrac{\mathbf{u}_i^\top \mathbf{y}}{d_k}\mathbf{v}_{ik}
$$
(d) We notice that the parameters for the linear regression obtained with the $q$-top principal components is a truncated version of the original least squares parameters. We observe that the terms of the sum depending of small singular values have been discarded.