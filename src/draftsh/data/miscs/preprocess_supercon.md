
# Xu 2025's preprocess  
follow 2.1 Data collection and cleaning
* SuperCon version:  
    > over '14794'  
    * I don't know which version of [SuperCon](https://mdr.nims.go.jp/collections/5712mb227) employed
    * note that Stanev et al(2018) used over 16k compositions. I will use that
1. Absence of $T_c$.   
    > ... cleaned out 3052 data points  
* Stanev et al. 2018 classified non-Sc observed instances to train classifier
2. ambiguous composition.   
    > ... As a result, we cleaned 24 data points of this type
3. Data redundancy
    > ... Firstly, the identical composition has different Tc values.
    > ... Secondly, the proportions of elements in the material are adjusted in multiples, or the order of elements is modified.
    > ... Nb3Pt1 exhibits Tc of both 8.1 K and 11 K  
    > ... cleaned out 6505 data points of this type.
    > ... For example, $Ga_7Pt_3(C_{1.35}Ti_{0.1}Y_{0.9})$ and $Ga_{0.7}Pt_{0.3}(Y_{1.8}Ti_{0.2}C_{2.7})$ ...the same material
 
* final data points count:
    > we obtained 5213 entries

* final notes: 
    > therefore, decided not to include high-temperature superconductors in this study.
    * I cannot figure out the criteria of high-T_c.