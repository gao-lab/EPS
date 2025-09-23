# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: R_systerm
#     language: R
#     name: r_systerm
# ---

# %% vscode={"languageId": "r"}
library(clusterProfiler)
library(DOSE)
library(org.Hs.eg.db)
library(enrichplot)

# %% [markdown]
# ## Top genes

# %% vscode={"languageId": "r"}
gene_df <- read.csv('./results/per_gene_mean_median.csv')
# first col to index
rownames(gene_df) <- gene_df[,1]
gene_df <- gene_df[,-1]
# sort by mean
gene_df <- gene_df[order(-gene_df$mean),]

# get mean > 0.1
sub_df = gene_df[gene_df$mean > 0.25,]
tail(sub_df,3)
gene_list <- rownames(sub_df)
length(gene_list)

# %% vscode={"languageId": "r"}
GO_database <- 'org.Hs.eg.db'
KEGG_database <- 'hsa'

# %% vscode={"languageId": "r"}
gene <- bitr(gene_list,fromType = 'SYMBOL',toType = 'ENTREZID',OrgDb = GO_database)
head(gene,3)

# %% vscode={"languageId": "r"}
GO <- enrichGO(gene$ENTREZID, 
             ont ="ALL", 
             keyType = "ENTREZID",
             pvalueCutoff = 0.05,
             qvalueCutoff = 0.05,
             OrgDb = GO_database)

# %% vscode={"languageId": "r"}
KEGG<-enrichKEGG(gene$ENTREZID,
                 organism = KEGG_database,
                 pvalueCutoff = 0.05,
                 qvalueCutoff = 0.05)

# %% vscode={"languageId": "r"}
options(repr.plot.width = 7, repr.plot.height = 10)
barplot(GO, split="ONTOLOGY", showCategory=5 )+facet_grid(ONTOLOGY~., scale="free")

options(repr.plot.width = 8, repr.plot.height = 6)
barplot(KEGG,showCategory = 40,title = 'KEGG Pathway')

# %% vscode={"languageId": "r"}
dotplot(GO, split="ONTOLOGY")+facet_grid(ONTOLOGY~., scale="free")
dotplot(KEGG)

# %% vscode={"languageId": "r"}
GO_CC <- enrichGO(gene$ENTREZID, 
             ont ="CC", 
             keyType = "ENTREZID",
             pvalueCutoff = 0.05,
             qvalueCutoff = 0.05,
             OrgDb = GO_database)

GO_CC2 <- setReadable(GO_CC, 'org.Hs.eg.db', 'ENTREZID')
options(repr.plot.width = 9, repr.plot.height = 9)

cnetplot(GO_CC2, categorySize = "pvalue", showCategory = 10)

# %% vscode={"languageId": "r"}
GO2 <- setReadable(GO, 'org.Hs.eg.db', 'ENTREZID')
KEGG2 <- setReadable(KEGG, 'org.Hs.eg.db', 'ENTREZID')
options(repr.plot.width = 9, repr.plot.height = 9)
enrichplot::cnetplot(GO2, circular=TRUE, colorEdge = TRUE)
enrichplot::cnetplot(KEGG2, circular=FALSE, colorEdge = TRUE)

# %% [markdown]
# ## Tail genes

# %% vscode={"languageId": "r"}
gene_df <- read.csv('./results/per_gene_mean_median.csv')
# first col to index
rownames(gene_df) <- gene_df[,1]
gene_df <- gene_df[,-1]
# sort by mean
gene_df <- gene_df[order(-gene_df$mean),]


sub_df = gene_df[gene_df$mean < 0.05,]
tail(sub_df,3)
gene_list <- rownames(sub_df)
length(gene_list)

# %% vscode={"languageId": "r"}
GO_database <- 'org.Hs.eg.db'
KEGG_database <- 'hsa'

# %% vscode={"languageId": "r"}
gene <- bitr(gene_list,fromType = 'SYMBOL',toType = 'ENTREZID',OrgDb = GO_database)
head(gene,3)

# %% vscode={"languageId": "r"}
GO <- enrichGO(gene$ENTREZID, 
             ont ="ALL", 
             keyType = "ENTREZID",
             pvalueCutoff = 0.05,
             qvalueCutoff = 0.05,
             OrgDb = GO_database)

# %% vscode={"languageId": "r"}
KEGG<-enrichKEGG(gene$ENTREZID,
                 organism = KEGG_database,
                 pvalueCutoff = 0.05,
                 qvalueCutoff = 0.05)

# %% vscode={"languageId": "r"}
options(repr.plot.width = 7, repr.plot.height = 10)
barplot(GO, split="ONTOLOGY")+facet_grid(ONTOLOGY~., scale="free")

options(repr.plot.width = 8, repr.plot.height = 6)
barplot(KEGG,showCategory = 40,title = 'KEGG Pathway')

# %% vscode={"languageId": "r"}
dotplot(GO, split="ONTOLOGY")+facet_grid(ONTOLOGY~., scale="free")
dotplot(KEGG)

# %% vscode={"languageId": "r"}
GO_CC <- enrichGO(gene$ENTREZID, 
             ont ="CC", 
             keyType = "ENTREZID",
             pvalueCutoff = 0.05,
             qvalueCutoff = 0.05,
             OrgDb = GO_database)

GO_CC2 <- setReadable(GO_CC, 'org.Hs.eg.db', 'ENTREZID')
options(repr.plot.width = 9, repr.plot.height = 9)

cnetplot(GO_CC2, categorySize = "pvalue", showCategory = 10)

# %% vscode={"languageId": "r"}
GO2 <- setReadable(GO, 'org.Hs.eg.db', 'ENTREZID')
KEGG2 <- setReadable(KEGG, 'org.Hs.eg.db', 'ENTREZID')
options(repr.plot.width = 9, repr.plot.height = 9)
enrichplot::cnetplot(GO2, circular=TRUE, colorEdge = TRUE)
enrichplot::cnetplot(KEGG2, circular=FALSE, colorEdge = TRUE)
