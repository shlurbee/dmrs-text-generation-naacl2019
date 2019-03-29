# Examples of how to invoke eval script for different models and test/dev datasets

bash scripts/eval.sh models/news_gold_acc_77.03_ppl_9.86_e30.pt data_ws/test/test data/anon-replacements.json 2
bash scripts/eval.sh models/news_gold_acc_77.03_ppl_9.86_e30.pt data_wsj/test/test data/anon-replacements.json 2
bash scripts/eval.sh models/news_gold_silver_acc_91.27_ppl_1.78_e30.pt data_ws/test/test data/anon-replacements.json 2
bash scripts/eval.sh models/news_gold_silver_acc_91.27_ppl_1.78_e30.pt data_wsj/test/test data/anon-replacements.json 2
bash scripts/eval.sh models/news_gold_acc_77.03_ppl_9.86_e30.pt data_brown/test/test data/anon-replacements.json 2
bash scripts/eval.sh models/news_gold_silver_acc_91.27_ppl_1.78_e30.pt data_brown/test/test data/anon-replacements.json 2
bash scripts/eval.sh models/data_vs1_acc_91.07_ppl_1.88_e30.pt data/dev/dev data/anon-replacements.json 2
