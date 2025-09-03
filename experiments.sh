# python main.py --problem "Maximum Coverage" --budget 100 --dataset "HK" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "Wiki" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "Deezer" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "Slashdot" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "Twitter" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "DBLP" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "YouTube" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "Skitter" --iterations 10
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Facebook
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Wiki
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Deezer
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Slashdot
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Twitter
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset DBLP
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset YouTube
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Skitter


python finetune.py --problem "Maximum Cut" --dataset Facebook
python finetune.py --problem "Maximum Cut" --dataset Wiki
python finetune.py --problem "Maximum Cut" --dataset Deezer
python finetune.py --problem "Maximum Cut" --dataset Slashdot
python finetune.py --problem "Maximum Cut" --dataset Twitter
python finetune.py --problem "Maximum Cut" --dataset DBLP
python finetune.py --problem "Maximum Cut" --dataset YouTube
python finetune.py --problem "Maximum Cut" --dataset Skitter

python test.py --problem "Maximum Cut" --train_dataset Facebook --test_dataset Facebook
python test.py --problem "Maximum Cut" --train_dataset Wiki --test_dataset Wiki
python test.py --problem "Maximum Cut" --train_dataset Deezer --test_dataset Deezer
python test.py --problem "Maximum Cut" --train_dataset Slashdot --test_dataset Slashdot
python test.py --problem "Maximum Cut" --train_dataset Twitter --test_dataset Twitter
python test.py --problem "Maximum Cut" --train_dataset DBLP --test_dataset DBLP
python test.py --problem "Maximum Cut" --train_dataset YouTube --test_dataset YouTube
python test.py --problem "Maximum Cut" --train_dataset Skitter --test_dataset Skitter


# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Facebook
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Wiki
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Deezer
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Slashdot
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Twitter
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset DBLP
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset YouTube
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Skitter

# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Facebook
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Wiki
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Deezer
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Slashdot
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Twitter
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset DBLP
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset YouTube
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Skitter