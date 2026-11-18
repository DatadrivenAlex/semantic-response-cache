from semantic_cache.cache import ResponseCache


def main():
    cache = ResponseCache()
    cache.put(scope="demo", model="local-v1", prompt="what is rate limiting",
              answer="Limits requests over time.", ttl=60)
    print("hit:", cache.lookup(scope="demo", model="local-v1", prompt="what is rate limiting"))
    print("miss:", cache.lookup(scope="demo", model="local-v1", prompt="what is docker"))
    cache.close()


if __name__ == "__main__":
    main()
