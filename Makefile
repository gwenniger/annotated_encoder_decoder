notebook: annotated_encoder_decoder.py
	jupytext --to ipynb annotated_encoder_decoder.py

py: annotated_encoder_decoder.ipynb
	jupytext --to py:percent annotated_encoder_decoder.ipynb

annotated_encoder_decoder.ipynb: annotated_encoder_decoder.py
	jupytext --to ipynb annotated_encoder_decoder.py

execute: annotated_encoder_decoder.py
	jupytext --execute --to ipynb annotated_encoder_decoder.py

html: annotated_encoder_decoder.ipynb
	jupytext --execute --to ipynb annotated_encoder_decoder.py
	jupyter nbconvert --to html annotated_encoder_decoder.ipynb

annotated_encoder_decoder.md: annotated_encoder_decoder.ipynb
	jupyter nbconvert --to markdown  --execute annotated_encoder_decoder.ipynb

blog: annotated_encoder_decoder.md
	pandoc docs/header-includes.yaml annotated_encoder_decoder.md  --katex=/usr/local/lib/node_modules/katex/dist/ --output=docs/index.html --to=html5 --css=docs/github.min.css --css=docs/tufte.css --no-highlight --self-contained --metadata pagetitle="The Annotated Transformer" --resource-path=/home/srush/Projects/annotated-transformer/ --indented-code-classes=nohighlight



flake: annotated_encoder_decoder.ipynb
	flake8 --show-source annotated_encoder_decoder.py

black: annotated_encoder_decoder.ipynb
	black --line-length 79 annotated_encoder_decoder.py

clean: 
	rm -f annotated_encoder_decoder.ipynb

# see README.md - IWSLT needs to be downloaded manually to obtain 2016-01.tgz
move-dataset:
	mkdir -p ~/.torchtext/cache/IWSLT2016
	cp 2016-01.tgz ~/.torchtext/cache/IWSLT2016/.

setup: move-dataset
	pip install -r requirements.txt
