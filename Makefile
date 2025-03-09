dev-docs:
	uv run mkdocs serve -w deeplocalizer

dev:
	uv run deeplocalizer/deeplocalizer.py

publish:
	rm -fr dist/
	rm -fr deeplocalizer.egg-info/

	uv build
	uv publish --token $(TOKEN) 

