"""Tavily 検索ツール。

import 時に TAVILY_API_KEY が必要なため、Settings() を先にロードして
環境変数をセットする必要がある。
"""

from langchain_tavily import TavilySearch

from .settings import Settings

# 環境変数をセット（TavilySearch が TAVILY_API_KEY を参照するため）
Settings()

search_tool = TavilySearch(max_results=5)
