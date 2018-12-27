rm -rf test
git clone https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/d2l-ai/notebooks.git test
cd test
echo "test" >>README.md
git commit -am "upload"
git push
