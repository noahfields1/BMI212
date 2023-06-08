#Noah wrote this code

#Extracting SNP Data for Controls
#bcftools concat -a --regions chr1:119975336,chr2:226229029,chr2:60357684,chr2:43505684,chr3:185793899,chr3:12351626,chr3:64726228,chr4:6291188,chr5:77129124,chr6:20660803,chr7:28140937,chr7:130782095,chr8:117173494,chr8:94948283,chr9:22134095,chr9:79337213,chr10:112998590,chr10:92703125,chr10:12286011,chr11:2837316,chr11:2670241,chr11:17388025,chr11:72722053,chr12:71269322,chr12:121022883,chr12:65781114,chr15:90978107,chr15:80139880,chr16:53786615 dataset_49623708_vcfs/*.vcf.gz -o dataset_49623708_vcfs/dataset_49623708_merged_regions.vcf.gz

#plink --vcf dataset_49623708_vcfs/dataset_49623708_merged_regions.vcf.gz --make-bed --out dataset_49623708_plink

#Extracting SNP Data for Cases
bcftools concat -a --regions chr1:119975336,chr2:226229029,chr2:60357684,chr2:43505684,chr3:185793899,chr3:12351626,chr3:64726228,chr4:6291188,chr5:77129124,chr6:20660803,chr7:28140937,chr7:130782095,chr8:117173494,chr8:94948283,chr9:22134095,chr9:79337213,chr10:112998590,chr10:92703125,chr10:12286011,chr11:2837316,chr11:2670241,chr11:17388025,chr11:72722053,chr12:71269322,chr12:121022883,chr12:65781114,chr15:90978107,chr15:80139880,chr16:53786615 dataset_04669194_vcfs/*.vcf.gz -o dataset_04669194_vcfs/dataset_04669194_merged_regions.vcf.gz

plink --vcf dataset_04669194_vcfs/dataset_04669194_merged_regions.vcf.gz --make-bed --out dataset_04669194_plink
