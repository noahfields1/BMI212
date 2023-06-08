import pandas as pd

def main():
    data = pd.read_csv('T2D_df.csv')
    data = data.replace(2, 1)
    filtered_df = data[data['race'] == '0tive Hawaiian or Other Pacific Islander']
    t2d = filtered_df['T2D_Status']
    count = 0
    for val in t2d:
        count+=1
        if val == 1:
            print("hi")
    print(count)


# Run the main function
if __name__ == "__main__":
    main()
