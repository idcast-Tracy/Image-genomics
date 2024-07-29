cat("\014"); rm(list = ls()); options(warn = -1); options(digits=3) 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(org.Hs.eg.db);library(clusterProfiler);library(enrichplot);
library(ggplot2);library(DESeq2);library(openxlsx);library(biomaRt);library(progress)
library(enrichplot);library(latex2exp);library(ggVolcano);library(ggrepel);library(GseaVis)
# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("clusterProfiler")
# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("DESeq2")
# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("biomaRt")
# install.packages("devtools")
# devtools::install_github("BioSenior/ggVolcano")
library(tidyverse);library(msigdbr);library(clusterProfiler);library(enrichplot);
library(RColorBrewer);library(ggrepel);library(ggplot2);library(aplot);
pvalueFilter=0.05      
qvalueFilter=0.05     

#--------------------------DESeq2------------
data = read.csv("TCGA-count113.csv",row.names = 1)
# load("TCGA-UCS_mrna_expr_counts.rdata");data = mrna_expr_counts
exprSet <- data

group <- read.csv("TCGA_group.csv",col.names = T);colnames(group) = "label"
group <- as.factor(group$label);str(group)

Data <- data.frame(row.names = colnames(exprSet), 
                   group = group)

dds <- DESeqDataSetFromMatrix(countData = exprSet,
                              colData = Data,
                              design = ~ group)

dds2 <- DESeq(dds)  
tmp <- results(dds2,contrast = c("group",1,0)) 
DEG_DESeq2 <- as.data.frame(tmp);DEG_DESeq2 <- na.omit(DEG_DESeq2)

FC <- 4;Padj <- 0.05
DEG_DESeq2$Significant <- "normal"
up <- intersect(which(DEG_DESeq2$log2FoldChange > log2(FC) ),
                which(DEG_DESeq2$padj < Padj))

down <- intersect(which(DEG_DESeq2$log2FoldChange < (-log2(FC))),
                  which(DEG_DESeq2$padj < Padj))
DEG_DESeq2$Significant[up] <- "up"
DEG_DESeq2$Significant[down] <- "down"
table(DEG_DESeq2$Significant)

up_gene = DEG_DESeq2[which(DEG_DESeq2$Significant == "up"),]
down_gene = DEG_DESeq2[which(DEG_DESeq2$Significant == "down"),]


#----------------------- GSEA-----------------------
data <- DEG_DESeq2;dim(data)
data_sort <- data %>%   
  arrange(desc(log2FoldChange))
gene_list <- data_sort$log2FoldChange
names(gene_list) <- rownames(data_sort)

res <- gseGO(
  gene_list,    
  ont = "BP",    
  OrgDb = org.Hs.eg.db,   
  keyType = "SYMBOL",    
  pvalueCutoff = 0.05,
  pAdjustMethod = "BH",   
  minGSSize    = 100,
  maxGSSize    = 500,
)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
dir.create("GSEAPlot");
N = length(res@result[["ID"]]);pb <- progress_bar$new(total=N)   

for (i in 246:N) {
  pb$tick()
  setwd(paste0(getwd(),"/GSEAPlot"))
  pdf_file <- paste0(res$Description[i], "-GSEA.pdf")
  pdf(pdf_file, wi=9, h=7)
  gsdata <- gsInfo(res, geneSetID = res$ID[i])
  gsdata1 <- gsdata %>%
    mutate("gene_name" = names(res@geneList)) %>%
    filter(position == 1)

  pseg <- ggplot(gsdata,aes(x = x,y = runningScore)) +
    geom_segment(data = gsdata1,
                 aes(x = x,xend = x,y = 0,yend = 1),
                 color = 'black',show.legend = F) +
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(expand = c(0,0)) +
    theme_bw(base_size = 14) +
    theme(axis.ticks = element_blank(),
          axis.text = element_blank(),
          axis.title.y = element_blank(),
          panel.grid = element_blank(),
          axis.line.x = element_blank(),
          plot.margin = margin(t = 0,r = .2,b = .2,l = .2,unit = "cm")) +
    xlab('Rank in Ordered Dataset')

  v <- seq(1, sum(gsdata$position), length.out = 9)
  inv <- findInterval(rev(cumsum(gsdata$position)), v)
  if (min(inv) == 0) inv <- inv + 1

  col <- c(rev(brewer.pal(5, "Blues")), brewer.pal(5, "Reds"))

  # ymin <- min(p2$data$ymin)
  # yy <- max(p2$data$ymax - p2$data$ymin) * .3
  ymin <- 0
  yy <- 0.3
  xmin <- which(!duplicated(inv))
  xmax <- xmin + as.numeric(table(inv)[as.character(unique(inv))])
  d <- data.frame(ymin = ymin, ymax = yy,
                  xmin = xmin,
                  xmax = xmax,
                  col = col[unique(inv)])

  pseg_ht <- pseg + geom_rect(
    aes_(xmin = ~xmin,xmax = ~xmax,
         ymin = ~ymin,ymax = ~ymax,
         fill = ~I(col)),
    data = d,
    alpha = 0.8,
    inherit.aes = FALSE)

  panother <- ggplot(gsdata,aes(x = x,y = runningScore,color = runningScore)) +
    geom_hline(yintercept = 0,size = 0.8,color = 'black',
               lty = 'dashed') +
    geom_point() +
    geom_line() +
    geom_segment(data = gsdata1,aes(xend = x,yend = 0)) +
    theme_bw(base_size = 14) +
    # scale_color_gradient(low = '#336699',high = '#993399') +
    scale_color_gradient2(low = '#336699',mid = 'white',high = '#993399',midpoint = 0.2) +
    scale_x_continuous(expand = c(0,0)) +
    theme(legend.position = 'none',
          axis.ticks.x = element_blank(),
          axis.text.x = element_blank(),
          axis.line.x = element_blank(),
          axis.title.x = element_blank(),
          legend.background = element_rect(fill = "transparent"),
          plot.margin = margin(t = .2,r = .2, b = 0,l = .2,unit = "cm"),
          plot.title = element_text(hjust = 0.5, size = 25)) +
    ylab('Enrichment Score') +
    ggtitle(res@result[["Description"]][i]) 
  panother_label <- panother
  color <- rev(colorRampPalette(c("#336699","white", "#993399"))(10))

  ht <- ggplot(gsdata,aes(x = x,y = runningScore)) +
    geom_rect(aes_(xmin = ~xmin,xmax = ~xmax,
                   ymin = ~ymin,ymax = ~ymax,
                   fill = ~I(color)),
              data = d,
              alpha = 0.8,
              inherit.aes = FALSE) +
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(expand = c(0,0)) +
    theme_bw(base_size = 14) +
    theme(panel.grid = element_blank(),
          axis.ticks = element_blank(),
          axis.text = element_blank(),
          axis.title = element_blank(),
          plot.margin = margin(t = 0,r = .2, b = .2,l = .2,unit = "cm"))

    # combine
    a <- aplot::plot_list(gglist = list(panother,ht),
                     ncol = 1, heights = c(0.9,0.1))
    print(a)
    dev.off()
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}




