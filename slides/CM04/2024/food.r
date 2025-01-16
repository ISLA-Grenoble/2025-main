load('food.rda')
df = foodOfTheWorld$compact.CT

filename = 'biplot-food-01.pdf'
pdf(filename, height=9.9, width=16.8)
pca.food = prcomp(df, center=TRUE)
biplot(pca.food)
dev.off()

filename = 'biplot-food-02.pdf'
pdf(filename, height=9.9, width=16.8)
pca.food = prcomp(df, center=TRUE, scale=TRUE)
biplot(pca.food)
dev.off()

filename = 'biplot-food-03.pdf'
pdf(filename, height=9.9, width=16.8)
df_norm = df /rowSums(df)
pca.food = prcomp(df_norm, center=TRUE)
biplot(pca.food)
dev.off()