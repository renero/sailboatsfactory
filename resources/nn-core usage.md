# nn-core usage

## data

How do I move (stepwise) from raw CSV file containing basic OHLC tick information, into training files for the different neural networks.

### CSE

Input data is CSV. We want it convert it into CSE (CandleStick Encoded).

    csv -> cse
    
### One Hot Encoding

From here, we want `.cse` to be converted to binary representation for deep learning training.

    cse -> ohe
    
This is a typical _one hot encoding_ (`ohe`) that depends on the dictionary of categorical states at which a candlestick body or movement can be. This is tipically 26 for the body, and 10 for the movement.

The `ohe` files must then be splitted into body and movement files.

    cse
     ├─> ohb
     └─> ohm
 
These are the extension names given to both filenames:
 
   - `ohb`: OneHot Body
   - `ohm`: OneHot Movement

### Windowed supervised timeseries

These files also need to be converted into a **supervised** format, using a specific **window size**.

Window size is the number of elements in a timely ordered series of tick prices that are considered relevant for the tratining of the prediction system. This can range from 3 to several dozens... it will depend on the timerange to be considered relevant to learn how to generalize a prediction.

The extension of the file will not give information about the window size used, so it will have to be extracted from reading it.

Supervised means that the file will add the expected response to be produced after each windowed series of ticks.

    ohb -> sohb
    ohm -> sohm
    
So, the entire chain of conversions can be outlined as (both directions):

    csv <-> cse <-> ohe
                     ├─> ohb <-> sohb 
                     └─> ohm <-> sohm