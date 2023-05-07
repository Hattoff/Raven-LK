TODO:
1. Need to update conversation manager to load all messages for the user and load as many past conversations as possible for prompting.
    I tried to remedy this by modifying how the states are saved but that is not sufficient; especially when user or Raven has a particularly
    large messages which trigger compressions with a small number of memories in the cache.
2. A class which manages all transactions. I am constantly finding myself struggling to recover from a mid-response crash because a
    lot of information has been changed but the full response was not generated. This transaction manager would prevent state saves and indexing
    until everthing else is complete, or it will save the intermediate states so that they can continue after recovering from the interruption.
3. Generalize the prompting or sub-prompting mechanic a bit more. Right now it feels like spaghet. Get some async prompts in place to quicken some of these processes.
4. Update the user interface to handle async updates to the debug window.
5. Transition from creating text documents to store this information to a SQL Server.


SQLite 3:
    Tables:
        Memories
            ID
            Speaker
            Depth
            Content
            Content Tokens
            Summary
            Summary Tokens
            Episodic Parent ID
            Past Sibling ID
            Next Sibling ID
            Theme Links [List]
            Total Theme Count
            Create Date
            Modify Date
        Theme Links
            ID
            Memory ID
            Theme ID
            Weight
            Recurrence
            Depth
            Cooldown
            Create Date
            Modify Date
        Themes
            ID
            Themes [List]
            Theme History [List]
            Create Date
            Modify Date
        Theme History
            ID
            Theme ID
            Phrase
            Iteration
            Similarity
            Create Date
            Modify Date


Could you write a create statement for a table called Memories with these columns? The ID field is a required text field and is the primary key, all other fields can be nullable. All fields are TEXT data types unless otherwise stated in [square brackets]. All field names should be lower case and use a _ underscore instead of a space.

            ID
            Speaker
            Depth [INTEGER]
            Content
            Content Tokens [INTEGER]
            Summary
            Summary Tokens [INTEGER]
            Episodic Parent ID
            Past Sibling ID
            Next Sibling ID
            Theme Links
            Total Themes [INTEGER]
            Create Date
            Modify Date

Could you write a create statement for a table called Theme Links with these columns? The ID field is a required text field and is the primary key, all other fields can be nullable. All fields are TEXT data types unless otherwise stated in [square brackets]. All field names should be lower case and use a _ underscore instead of a space. It should be in the format that Python can execute.

    ID
    Memory ID
    Theme ID
    Weight [REAL]
    Recurrence [INTEGER]
    Depth  [INTEGER]
    Cooldown  [INTEGER]
    Create Date
    Modify Date


Could you write a create statement for a table called Themes with these columns? The ID field is a required text field and is the primary key, all other fields can be nullable. All fields are TEXT data types unless otherwise stated in [square brackets]. All field names should be lower case and use a _ underscore instead of a space. It should be in the format that Python can execute.
    ID
    themes
    Theme History
    Create Date
    Modify Date


Could you write a create statement for a table called Theme History with these columns? The ID field is a required text field and is the primary key, all other fields can be nullable. All fields are TEXT data types unless otherwise stated in [square brackets]. All field names should be lower case and use a _ underscore instead of a space. It should be in the format that Python can execute.
    ID
    Theme ID
    Phrase
    Iteration [INTEGER]
    Similarity [REAL]
    Create Date
    Modify Date
