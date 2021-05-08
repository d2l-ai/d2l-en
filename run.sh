sed -i .bak 's/eval_notebook = True/eval_notebook = False/' config.ini
for i in {1..100}; do clear; d2lbook build slides --tab pytorch; sleep 10; done
sed -i .bak 's/eval_notebook = False/eval_notebook = True/' config.ini