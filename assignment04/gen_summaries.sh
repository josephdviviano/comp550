#!/bin/bash

mkdir outputs

./sumbasic.py orig docs/doc1-* > outputs/doc1_orig_summary.txt
./sumbasic.py orig docs/doc2-* > outputs/doc2_orig_summary.txt
./sumbasic.py orig docs/doc3-* > outputs/doc3_orig_summary.txt
./sumbasic.py orig docs/doc4-* > outputs/doc4_orig_summary.txt

./sumbasic.py best-avg docs/doc1-* > outputs/doc1_best-avg_summary.txt
./sumbasic.py best-avg docs/doc2-* > outputs/doc2_best-avg_summary.txt
./sumbasic.py best-avg docs/doc3-* > outputs/doc3_best-avg_summary.txt
./sumbasic.py best-avg docs/doc4-* > outputs/doc4_best-avg_summary.txt

./sumbasic.py simplified docs/doc1-* > outputs/doc1_simplified_summary.txt
./sumbasic.py simplified docs/doc2-* > outputs/doc2_simplified_summary.txt
./sumbasic.py simplified docs/doc3-* > outputs/doc3_simplified_summary.txt
./sumbasic.py simplified docs/doc4-* > outputs/doc4_simplified_summary.txt

./sumbasic.py leading docs/doc1-* > outputs/doc1_leading_summary.txt
./sumbasic.py leading docs/doc2-* > outputs/doc2_leading_summary.txt
./sumbasic.py leading docs/doc3-* > outputs/doc3_leading_summary.txt
./sumbasic.py leading docs/doc4-* > outputs/doc4_leading_summary.txt

