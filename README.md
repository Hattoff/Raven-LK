# Raven - Lore Keeper
Based on the work by [David Shapiro](https://github.com/daveshap), where this bot gets its name, Raven-LK is intended to help its user create a fictional world and populate it with believable and detailed elements.

Packages used:

- tiktoken
- pinecone
- openai
- configparser
- spellchecker (pyspellchecker)
- screeninfo
- various other bits that should already be installed

How to use:

- Run make_required_folders.py to get the folders you need in place.
- Make sure you have your key_openai.txt and key_pinecone.txt files in the api_keys folder.
- Check the config file under pinecone and set your region (environment) and index (vector database name) to match your settings. _**If you intend on using pinecone, set the config file's pinecone_indexing_enabled = True and function calls to pinecone will be not be skipped. It is False by default**_
- Run Raven-LK-py, it will create any folder structure you are missing which it is relying on.

Raven-LK is in early development but here is a short list of features I intend to implement:

* Query

  Search the local database for relevant information to answer any questions or inspire the writer with ideas for plot or character development.

* Create

  Generate story elements for the local database as needed.

* Modify

  Make changes to existing story elements entries as it deems necessary.

* Track

  Keep track of changes to story elements as you write and refine details.

* Identify

  Identify contradictions, inconsistancies, or generally undeveloped story elements.

* Maintain

  Maintain relationships between story elements so the writer can navigate increasingly complex story elements as their writing matures.
  
* Research

  Give the writer access to all information the base Large Language Model has to offer, allowing for quick fact checking (the writer should stil verify complex topics on thier own) or real-world insight.

* Inspire

  Interact with the writer to inspire and stir their creativity, offer suggestions to amplify their writings, or simply be an active listener the writer can bounce ideas off of and work through problems.

* Interplay

  A bit of a stretch goal, this feature would allow the writer to establish a scenerio with their story elements and set them lose. Raven-LK would interact with itself in the personas of characters, creatures, items, and locations to drive a dynamic scene. Allowing the writer to observe how their world could behave. Such role playing affords the writer an outside perspective on their creations and encourages them to explore more complex characters, relationships, environments, and creatures. These details not only better instruct Raven-LK on how to role play these elements, but also gives the writer insight into potentially contradictive bevahior they may have imagined taking place but Raven-LK finds no motivation to act out.
