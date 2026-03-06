#!/bin/bash
# Add all files under 50MB
find . -not -path './.git/*' -type f -size -50M | while read f; do
    git add "$f"
done
git commit -m "Updated files"
git push origin rebuttal
