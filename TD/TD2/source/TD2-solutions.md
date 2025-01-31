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

(c) Matrix $\mathbf{Z}$ is the projection of the data matrix on its $q$-top principal components. Therefore, we have:
$$
\underbrace{\mathbf{Z}}_{n \times q} = \underbrace{\mathbf{X}}_{n \times p}\underbrace{\mathbf{V}_q}_{p \times q} = \underbrace{\mathbf{U}}_{n \times p}\underbrace{\mathbf{D}}_{p \times p}\underbrace{\mathbf{V}^\top}_{p \times p} \underbrace{\mathbf{V}_q}_{p \times q} = \mathbf{U}\mathbf{D}
\underbrace{\left[\begin{array}{c}
\mathbf{I}_q \\
\mathbf{0}_{(p - q) \times q}
\end{array}\right]}_{p \times q} = \underbrace{\mathbf{U}}_{n \times p}\underbrace{\mathbf{D}_q}_{p \times q} \quad \text{where} \quad \mathbf{D}_q = \left[\begin{array}{ccc}
d_1 & & \\
 & \ddots & \\
&  & d_q \\
 & \mathbf{0}_{(p-q) \times q} & 
\end{array}\right]
$$
We calculate the coefficients for the new regression model
$$
\hat{\gamma} = (\mathbf{Z}^\top\mathbf{Z})^{-1}\mathbf{Z}^\top \mathbf{y} = (\mathbf{D}^\top_q\mathbf{U}^\top \mathbf{U}\mathbf{D}_q)^{-1}\mathbf{D}_q\mathbf{U}^\top\mathbf{y} = \underbrace{(\mathbf{D}_q^\top\mathbf{D}_q)^{-1}}_{q \times q}\underbrace{\mathbf{D}^\top_q}_{q \times p}\underbrace{\mathbf{U}^\top}_{p \times n}\underbrace{\mathbf{y}}_{n \times 1} = \left[\begin{array}{c} \tfrac{1}{d_1}\mathbf{u}_1^\top\mathbf{y} \\ \vdots \\ \tfrac{1}{d_q}\mathbf{u}_q^\top\mathbf{y}\end{array}\right]
$$
where we note that the $\gamma$ coefficients can be calculated as if we had $q$ independent simple linear regressions. This is due to the diagonal shape of the matrix $\mathbf{Z}$ as per:
$$
\mathbf{Z}^\top \mathbf{Z} = \mathbf{D}^\top_q\mathbf{D}_q = \left[\begin{array}{ccc}
d_1 & & \\
 & \ddots & \\
&  & d_q
\end{array}\right]
$$
If we take it back to the original space, we get
$$
\hat{\beta}^{\text{PCR}} = \mathbf{V}_q \hat{\gamma} = \mathbf{V}_q \left[\begin{array}{c} \tfrac{1}{d_1}\mathbf{u}_1^\top\mathbf{y} \\ \vdots \\\tfrac{1}{d_q}\mathbf{u}_q^\top\mathbf{y}\end{array}\right] = \left[\begin{array}{ccc} \mathbf{v}_1 & \dots & \mathbf{v}_q\end{array}\right]\left[\begin{array}{c} \tfrac{1}{d_1}\mathbf{u}_1^\top\mathbf{y} \\ \vdots \\ \tfrac{1}{d_q}\mathbf{u}_q^\top\mathbf{y}\end{array}\right] \quad \text{and} \quad \hat{\beta}^{\text{PCR}}_i = \sum_{k = 1}^q \dfrac{\mathbf{u}_i^\top \mathbf{y}}{d_k}\mathbf{v}_{ik}
$$
(d) We notice that the parameters for the linear regression obtained with the $q$-top principal components is a truncated version of the original least squares parameters. We observe that the terms of the sum depending of small singular values have been discarded.