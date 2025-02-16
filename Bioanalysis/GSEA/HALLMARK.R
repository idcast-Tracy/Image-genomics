# -------------------------- GSEA --------------------------------------
cat("\014"); rm(list = ls()); options(stringsAsFactors = F); options(warn = -1)
library(limma); library(org.Hs.eg.db); library(clusterProfiler); library(enrichplot); library('ggplot2')
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# ----------------------------------------------------
# Risk = read.csv("Inter-new_Tr_Risk2.csv", row.names = 1); head(Risk); co=1
Risk = read.csv("TCGA_group.csv", row.names = 1); head(Risk); co=1
#               f2    f14     f15
# X11150187 0.0899 0.1820 0.01535

# -------------------------- ---------------------------
if (co==1){FPKM = as.matrix(read.csv("TCGA-count113.csv", row.names=1)); dim(FPKM)}
# if (co==2){FPKM = as.matrix(read.csv("Inter-te-ComBat_TME_Hub_GGH.csv", row.names=1)); dim(FPKM)}
FPKM[which((FPKM<0))]=0; min(FPKM)
#           X10807656 X10956556 X10920835
# A4GNT       5.28      5.06      6.56

# ----------- FPKM->TPM ---------------
fpkmToTpm <- function(fpkm){exp(log(fpkm+1) - log(sum(fpkm)+1) + log(1e6))}
TPM <- log2(apply(FPKM,2,fpkmToTpm) + 1); min(TPM)

# -------------------------- logFC -------------------------- 
unique(Risk['label'])
TPML=TPM[,row.names(Risk[which(Risk[,"label"]==1),])]; meanL=rowMeans(TPML)
TPMH=TPM[,row.names(Risk[which(Risk[,"label"]==0),])]; meanH=rowMeans(TPMH)
meanL[meanL<0.00001]=0.00001; meanH[meanH<0.00001]=0.00001
logFC=log2(meanH)-log2(meanL); logFC=sort(logFC,decreasing=T)
genes=names(logFC)


# ------------------ GSEA  -----------------------
gmt1=read.gmt("c2_c5_h.all.v7.5.1.symbols.gmt") 
Diff.path=GSEA(logFC, TERM2GENE=gmt1, pvalueCutoff = 1, minGSSize = 5, pAdjustMethod = "BH", maxGSSize = 500,seed = 60)
if (co==1){Diff.path = Diff.path; save(Diff.path, file='Diff.path.tr.Rdata')}
if (co==2){Diff.path = Diff.path; save(Diff.path, file='Diff.path.te.Rdata')}


# ----------------------------------------------------------- 
Diff.tab=as.data.frame(Diff.path)
Sig_path1 = Diff.path[which(abs(Diff.tab$NES)>1 & Diff.tab$p.adjust<0.25 & (Diff.tab$pvalue<0.05)), ]; dim(Sig_path1)
Sig_path2 = Diff.path[which(abs(Diff.tab$NES)>1 & Diff.tab$pvalue<0.1), ]; dim(Sig_path2)
if (co==1){Sig_path.tr = Sig_path2; save(Sig_path.tr, file='Sig_path.tr.Rdata')}
if (co==2){Sig_path.te = Sig_path2; save(Sig_path.te, file='Sig_path.te.Rdata')}



# =================================================================================
cat("\014"); rm(list = ls())
load('Sig_path.tr.Rdata'); load('Sig_path.te.Rdata')
# ----- p<0.1 -----------
Inter1 = intersect(row.names(Sig_path.tr[which(Sig_path.tr$pvalue<0.1), ]),
                   row.names(Sig_path.te[which(Sig_path.te$pvalue<0.1), ]))
Sig_path.tr = Sig_path.tr[Inter1, ]; Sig_path.te = Sig_path.te[Inter1, ]

# ----- NES-----------
Inter2 = row.names(Sig_path.tr[which(Sig_path.tr['NES'] * Sig_path.te['NES'] > 0), ]); length(Inter2)

write.csv(Sig_path.tr[Inter2, ], 'Sig2_path.tr.csv')
write.csv(Sig_path.te[Inter2, ], 'Sig2_path.te.csv')


# =============================================================================
load('Diff.path.tr.Rdata'); co='tr_'
# load('Diff.path.te.Rdata'); co='te_'

# --- UP ---- 
pdf(paste0(co, "GSEA.UP.pdf"), width=9.5, height=9)
ID = c('GOBP_REGULATION_OF_CELL_CYCLE_PROCESS', 'GOBP_MITOTIC_SPINDLE_ORGANIZATION', 'GOBP_CELL_CYCLE_G2_M_PHASE_TRANSITION', 'GOBP_MITOTIC_CELL_CYCLE', 'GOBP_MICROTUBULE_CYTOSKELETON_ORGANIZATION_INVOLVED_IN_MITOSIS')
gseaplot2(Diff.path, geneSetID=ID,base_size=17,subplots=c(1:3),
          color = c('#a21919','#f36058', '#ff850d', '#ff6941', '#ffc90d'), 
          title = "Up-regulation pathway in the high-risk group", pvalue_table = F)
dev.off()

# --- DOWN ----
pdf(paste0(co, "GSEA.DN.pdf"), width=9.5, height=9)
ID = c('GOBP_MACROPHAGE_MIGRATION', 'GOBP_MACROPHAGE_CHEMOTAXIS', 'GOBP_REGULATION_OF_MACROPHAGE_CHEMOTAXIS', 'GOBP_NEGATIVE_REGULATION_OF_T_CELL_APOPTOTIC_PROCESS', 'GOBP_POSITIVE_REGULATION_OF_NATURAL_KILLER_CELL_CHEMOTAXIS')
gseaplot2(Diff.path, geneSetID=ID, base_size=17, subplots=c(1:3), 
          color=c('#2377e9', '#317bdf', '#2b8cdf', '#94a0f8', '#a9ceed'),
          title = "Down-regulation pathway in the high-risk group", pvalue_table = F)
dev.off()
print('finish')






# =================================================================================
# ------------------------------ GSEA ------------------------------ 
# |NES|>1, Pvalue<0.05, FDR[P.adj]<0.25
gse_cut <- Diff.path[Diff.path$pvalue<0.05 & Diff.path$p.adjust<0.25 & abs(Diff.path$NES)>1]
gse_cut_dn <- gse_cut[gse_cut$NES < 0,]
gse_cut_up <- gse_cut[gse_cut$NES > 0,]

#
dn_gsea <- gse_cut_dn[tail(order(gse_cut_dn$NES, decreasing = T),10),]
up_gsea <- gse_cut_up[head(order(gse_cut_up$NES, decreasing = T),10),]
diff_gsea <- gse_cut[head(order(abs(gse_cut$NES),decreasing = T),10),]


# ------------------------------ GSEA ------------------------------
# 
up_gsea$Description; i = 2 
pdf(paste0("GSEA-up ", up_gsea$ID[i], ".pdf"), width=14, height=12)
gseaplot2(gse, up_gsea$ID[i], title = up_gsea$Description[i],
          base_size = 22, ES_geom = "line", color = "red",
          rel_heights = c(1.5, 0.5, 0.5), subplots = 1:3, pvalue_table = T)
# ggsave(gseap1, filename = 'GSEA_up_1.pdf', width =10, height =8)
dev.off()


print('finish')