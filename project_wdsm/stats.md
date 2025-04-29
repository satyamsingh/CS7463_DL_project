

# General Stats

Winner Statistics:
------------------
winner
model_b    24481
model_a    23958
Name: count, dtype: int64

Total samples: 48439
Win rates:
model_a: 49.5%
model_b: 50.5%


Top 10 languages by percentage:
------------------------------
English: 52.06% (17,699 samples)
Russian: 13.21% (4,491 samples)
Chinese: 8.98% (3,053 samples)
Vietnamese: 6.37% (2,165 samples)
German: 2.91% (988 samples)
Japanese: 2.38% (808 samples)
unknown: 2.33% (793 samples)
Korean: 2.13% (723 samples)
Spanish: 1.61% (548 samples)
French: 1.33% (453 samples)

Bottom 10 languages by percentage:
--------------------------------
Turkmen: 0.00% (1 samples)
Hausa: 0.00% (1 samples)
Zhuang: 0.00% (1 samples)
Tswana: 0.00% (1 samples)
Lingala: 0.00% (1 samples)
Scottish Gaelic: 0.00% (1 samples)
Telugu: 0.00% (1 samples)
Swati: 0.00% (1 samples)
Sindhi: 0.00% (1 samples)
Khmer: 0.00% (1 samples)



# Prompt Length Analysis:


Prompt length statistics by language (top 15):
                   mean  median  count
language                              
English     1040.287022   175.0  17699
Russian      912.594077   181.0   4491
Chinese      392.577465    54.0   3053
Vietnamese   520.037413   110.0   2165
German       610.283401   127.0    988
Japanese     288.806931    54.0    808
unknown     1579.281211   124.0    793
Korean       430.001383    43.0    723
Spanish      628.281022   201.0    548
French       900.185430   153.0    453
Portuguese   789.569270   202.0    397
Persian      696.414449   117.0    263
Italian      909.612717   172.0    173
Turkish      260.227273    80.0    154
Czech        555.309524   109.0    126



Prompt length percentiles:
10th percentile: 29.00
25th percentile: 58.00
50th percentile: 141.00
75th percentile: 473.00
90th percentile: 1887.00
95th percentile: 3972.00
99th percentile: 13860.13



Win rates by length bin:
winner       model_a   model_b
length_bin                    
0-50        0.493506  0.506494
51-100      0.496584  0.503416
101-200     0.497644  0.502356
201-500     0.483518  0.516482
501-1000    0.490573  0.509427
1001-5000   0.495765  0.504235
5000+       0.515766  0.484234


Combined model performance statistics (regardless of position):
                          model  win_rate  mean_prompt_length  count
0             yi-lightning-lite  0.509336          829.924881   2303
1             grok-2-2024-08-13  0.642857          879.575988   1974
2  llama-3.1-405b-instruct-bf16  0.543783          766.151427   2067
3          gemini-1.5-flash-002  0.545597         1002.751048   1908
4                 qwen-max-0919  0.520448          935.291864   2323
5        claude-3-opus-20240229  0.457368          923.504967   2416
6            gemini-1.5-pro-002  0.604800         1091.087200   2500
7    chatgpt-4o-latest-20240903  0.715391          926.281516   2586
8                qwen-plus-0828  0.408413          746.635250   1878
9       gemini-1.5-flash-8b-001  0.410661          888.632280   2026





# Reponse length

Response length statistics:
Model A: Mean = 2024.50, Median = 1607.00
Model B: Mean = 2018.88, Median = 1608.00

T-test for response length difference between models A and B:
t-statistic: 0.3881, p-value: 0.6980


T-test for response length difference between winners and losers:
t-statistic: 11.9019, p-value: 0.0000
Mean length of winning responses: 2107.89
Mean length of losing responses: 1935.49



Correlation analysis by language:
     language  correlation   p_value  sample_size
8     Spanish     0.130304  0.002240          548
3  Vietnamese     0.100298  0.000003         2165
9      French     0.089185  0.057863          453
6     unknown     0.086144  0.015245          793
4      German     0.066000  0.038062          988
1     Russian     0.058153  0.000096         4491
2     Chinese     0.041411  0.022128         3053
0     English     0.010347  0.168672        17699
5    Japanese    -0.031636  0.369142          808
7      Korean    -0.035220  0.344319          723

Interpretation guide:
Positive correlation: Model A is more likely to win when its response is longer relative to Model B
Negative correlation: Model A is more likely to win when its response is shorter relative to Model B
* p < 0.05, ** p < 0.01, *** p < 0.001

Languages with strongest correlations:
     language  correlation   p_value  sample_size
8     Spanish     0.130304  0.002240          548
3  Vietnamese     0.100298  0.000003         2165
9      French     0.089185  0.057863          453
6     unknown     0.086144  0.015245          793
4      German     0.066000  0.038062          988

Languages with strongest negative correlations:
   language  correlation   p_value  sample_size
1   Russian     0.058153  0.000096         4491
2   Chinese     0.041411  0.022128         3053
0   English     0.010347  0.168672        17699
5  Japanese    -0.031636  0.369142          808
7    Korean    -0.035220  0.344319          723



# Model Study

Top 10 models by win rate:
                         model  appearances  wins  win_rate
20  chatgpt-4o-latest-20240903         2586  1850  0.715391
14                  o1-preview         1647  1172  0.711597
23  chatgpt-4o-latest-20240808          936   633  0.676282
10                     o1-mini         1682  1127  0.670036
47     gemini-1.5-pro-exp-0827          681   451  0.662261
4            grok-2-2024-08-13         1974  1269  0.642857
43                yi-lightning         1622  1008  0.621455
19          gemini-1.5-pro-002         2500  1512  0.604800
44   gemini-1.5-flash-exp-0827          833   503  0.603842
22      gpt-4o-mini-2024-07-18         1220   733  0.600820

Bottom 10 models by win rate:
                          model  appearances  wins  win_rate
56  mixtral-8x22b-instruct-v0.1          108    39  0.361111
6          c4ai-aya-expanse-32b          316   111  0.351266
31            command-r-08-2024         1393   481  0.345298
9       claude-3-haiku-20240307         1222   413  0.337971
57               jamba-1.5-mini          276    88  0.318841
36         internlm2_5-20b-chat         1596   437  0.273810
28                gemma-2-2b-it         1689   435  0.257549
40     phi-3-medium-4k-instruct          151    32  0.211921
53        llama-3.2-3b-instruct         1343   262  0.195086
34        llama-3.2-1b-instruct         1455   179  0.123024




Most consistently strong models:
                        model_a  avg_win_rate  win_rate_std  num_opponents  \
18      gemini-1.5-pro-exp-0827      0.700204      0.163791             14   
45                   o1-preview      0.685823      0.167285             27   
3    chatgpt-4o-latest-20240903      0.684915      0.125986             41   
2    chatgpt-4o-latest-20240808      0.684060      0.131820             23   
15    gemini-1.5-flash-exp-0827      0.657333      0.196400             16   
44                      o1-mini      0.645265      0.162584             27   
29       gpt-4o-mini-2024-07-18      0.641221      0.178328             22   
17           gemini-1.5-pro-002      0.636432      0.117972             43   
30            grok-2-2024-08-13      0.636357      0.147095             36   
36  llama-3.1-405b-instruct-fp8      0.636297      0.162226             21   

    total_matches  
18            211  
45            699  
3            1201  
2             350  
15            255  
44            704  
29            425  
17           1291  
30            880  
36            417  



Most consistently weak models:
                    model_a  avg_win_rate  win_rate_std  num_opponents  \
53      reka-flash-20240904      0.409696      0.210373             26   
21            gemma-2-9b-it      0.380387      0.228154              9   
38    llama-3.1-8b-instruct      0.374363      0.199867             29   
8         command-r-08-2024      0.327405      0.161647             36   
6   claude-3-haiku-20240307      0.300258      0.176736             28   
32     internlm2_5-20b-chat      0.299081      0.169691             32   
20            gemma-2-2b-it      0.275824      0.131997             36   
11     gemini-1.5-flash-001      0.258986      0.104900              6   
42    llama-3.2-3b-instruct      0.182683      0.114172             31   
41    llama-3.2-1b-instruct      0.131493      0.110171             35   

    total_matches  
53            402  
21            126  
38            515  
8             642  
6             474  
32            786  
20            733  
11             81  
42            625  
41            703  



Models with highest dominance ratio (consistently beat other models):
                          model  total_opponents  dominated_opponents  \
45      gemini-1.5-pro-exp-0827               14                   12   
11   chatgpt-4o-latest-20240808               23                   19   
16   chatgpt-4o-latest-20240903               41                   31   
33                   o1-preview               27                   20   
17           gemini-1.5-pro-002               43                   29   
40    gemini-1.5-flash-exp-0827               16                   10   
39  llama-3.1-405b-instruct-fp8               21                   13   
43       gpt-4o-mini-2024-07-18               22                   13   
2             grok-2-2024-08-13               36                   20   
35                 yi-lightning               27                   15   

    dominance_ratio  avg_win_rate  
45         0.857143      0.700204  
11         0.826087      0.684060  
16         0.756098      0.684915  
33         0.740741      0.685823  
17         0.674419      0.636432  
40         0.625000      0.657333  
39         0.619048      0.636297  
43         0.590909      0.641221  
2          0.555556      0.636357  
35         0.555556      0.595134  