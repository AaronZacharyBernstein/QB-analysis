# 🏈QB ANALYSIS DETAILS🏈

## 💪1. Motivation behind the project and problem definition💪
   
    As I have been getting more familiar with the world of AI/machine learning, I wanted to build a project 
    where I could combine my passion for sports with my love for creation and engineering. 
    As a 49ers fan, I watched my team spend 3 first round picks to trade up and select Trey Lance, a dual-threat 
    FCS national champion who led his team to an undefeated season while contributing 42 total touchdowns with 0 interceptions.
    Fast forward to today, and Trey Lance is rotting as a backup in LA after a failed stint with the 49ers. 
    On the other hand, with the very last pick in the 2022 NFL draft, the 49ers selected Brock Purdy, a senior from 
    a middle-of-the-pack BIG12 school who had just come off a 4-8 season.
    Once again, fast forward to today, and this once-overlooked kid now has a 265 million dollar contract 
    and is thriving on one of the NFL’s most storied franchises.
    With so many other examples of situations almost identical to this, I set out to see if I could find a 
    true pattern behind what separates QBs who become NFL stars, and those who become labeled as draft busts.
    
## 📈2. Data collection📈
    Currently, every QB is evaluated by 21 features to produce a rating, each of which is listed below, and websites where I got the features from:
    
- **📏Height**: ESPN
- **⚖️Weight**: ESPN
- **📅Years starting**: Sports-reference cross validated with Google Gemini
- **🔢Draft position**: Sports-reference
- **⭐High school ranking**: 247sports
- **🏫School prestige**: My opinion based on info from Sports-reference
- **🤝Support cast**: My opinion based on info from Sports-reference
- **🏹Pass yards per game**: Sports-reference
- **💥Pass TD per game**: Sports-reference
- **☄️Attempts per game**: Sports-reference
- **🎯Completion %**: Sports-reference
- **⚠️Interceptions per game**: Sports-reference
- **⚙️Passing efficiency rating**: Sports-reference
- **👟Rush yards per game**: Sports-reference
- **💨Rush tds per game**: Sports-reference
- **⏱️40 yard dash**: MockDraftable, pro day statistics from Google search, or median if unavailable
- **🦘Vertical jump**: MockDraftable, pro day statistics from Google search, or median if unavailable
- **🖐️Hand size**: MockDraftable, pro day statistics from Google search, or median if unavailable
- **🧠Wonderlic/S2 score**: WonderlicTestPractice or Google search, left blank if no score available
- **📋Test score taken/released**: 1 if score was released, 0 if there is no score present
- **🏆Heisman**: Personal knowledge, 1 if they won, 0 if a player did not win the award
- **💎Final rating**: Ranked on a scale of 0-1 based on my personal knowledge and opinion of their NFL success

## ⚙️3. Python Libraries used⚙️
- _Pandas_: Used to read through the CSV file and create a data frame, filling in empty values and ensuring there are no errors with the input data
- _Scikit-learn_: Used to standardize all the data, making sure that no categories dominated others
- _PyTorch_: Created layers of a neural network and ran epochs, using tensors to calculate weights and build the model
- _Matplotlib_: Used to graph the training loss from epoch 1 to epoch 200 for me to see how the model is improving
