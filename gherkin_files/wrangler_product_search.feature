Feature: Search and add XL size shirt to cart on Wrangler website
  As a customer on wrangler.in
  I want to search for a solid blue shirt in XL size
  So that I can add it to my cart and purchase it

  @product_search @cart @xl_size
  Scenario: Search for solid blue shirt, verify XL size availability, add to cart and verify
    Given a user is on the URL as https://www.wrangler.in
    When the user clicks on the Search icon
    And the user enters "solid blue shirt" in the search bar
    And the user presses Enter to search
    Then search results should be displayed
    When the user clicks on the first matching product from the search results
    Then the Product Detail Page (PDP) should be displayed
    When the user checks for XL size availability on the PDP
    Then the XL size option should be available and selectable
    When the user selects the XL size
    And the user clicks on the Add to Cart button
    Then the product should be added to the cart successfully
    When the user navigates to the cart
    Then the cart should display the following:
      | Field        | Expected Value           |
      | Product Name | matches searched product |
      | Size         | XL                       |
      | Quantity     | 1                        |
