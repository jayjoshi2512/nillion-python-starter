from nada_dsl import *

def nada_main():
    # Define two parties
    party1 = Party(name="Party1")
    party2 = Party(name="Party2")
    
    # Define secret inputs for each party
    my_int1 = SecretInteger(Input(name="my_int1", party=party1))
    my_int2 = SecretInteger(Input(name="my_int2", party=party2))
    
    # Perform a secure comparison
    is_greater = my_int1 > my_int2
    
    # Use a conditional statement to select the maximum value securely
    max_value = If(is_greater, my_int1, my_int2)
    
    # Calculate the sum and difference of the two integers
    sum_value = my_int1 + my_int2
    diff_value = my_int1 - my_int2
    
    # Calculate the product and quotient
    product_value = my_int1 * my_int2
    quotient_value = my_int1 / my_int2
    
    # Combine all results into outputs
    return [
        Output(max_value, "max_output", [party1, party2]),
        Output(sum_value, "sum_output", [party1, party2]),
        Output(diff_value, "diff_output", [party1, party2]),
        Output(product_value, "product_output", [party1, party2]),
        Output(quotient_value, "quotient_output", [party1, party2])
    ]

