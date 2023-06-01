## Extract variants for each patient from the BED file using common SNPs from 'common_SNPs437.txt'
bedtools intersect -a <your_bed_file.bed> -b common_SNPs437.txt -wa -wb > output.bed
# After running the command, the output.bed file will contain the variants for each patient that align with the common SNPs from 'common_SNPs437.txt'
