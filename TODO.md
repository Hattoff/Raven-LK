TODO:
1. A class which manages all transactions. I am constantly finding myself struggling to recover from a mid-response crash because a
    lot of information has been changed but the full response was not generated. This transaction manager would prevent state saves and indexing
    until everthing else is complete, or it will save the intermediate states so that they can continue after recovering from the interruption.
2. Generalize the prompting or sub-prompting mechanic a bit more. Right now it feels like spaghet. Get some async prompts in place to quicken some of these processes.
3. Update the user interface to handle async updates to the debug window.
