@wrangler @search @shirts @size_check
Feature: Search for solid blue shirt and check XL availability on Wrangler.in
  As a customer of Wrangler India
  I want to search for a solid blue shirt
  So that I can check if size XL is available and add it to my cart

  Background:
    Given the user navigates to "https://www.wrangler.in/"
      # expected_visual_signals: ["Wrangler", "Search", "Cart"]
    And the user waits for the homepage to fully load
      # expected_visual_signals: ["MEN", "WOMEN", "NEW ARRIVALS", "DENIM"]
    And the user closes any promotional popup or banner if present
      # expected_visual_signals: ["close", "×", "X"]

  @shirts @size_check
  Scenario: Search for solid blue shirt, verify XL availability, and add to cart
    # Step 1: Search for product
    When the user clicks the search icon or focuses the search input
      # expected_visual_signals: ["Search", "magnifier"]
    And the user types "solid blue shirt" into the search field
      # expected_visual_signals: ["solid blue shirt"]
    And the user submits the search query
      # expected_visual_signals: ["Results", "products", "Found"]

    # Step 2: Select product
    Then search results page should show matching products
      # expected_visual_signals: ["Solid Blue Shirt", "₹", "Add to Cart"]
    When the user clicks on the first matching solid blue shirt product
      # expected_visual_signals: ["Product Details", "Description", "Size"]

    # Step 3: Check XL availability and Add to Cart
    Then the product detail page should display size options
      # expected_visual_signals: ["Size", "S", "M", "L", "XL", "Add to Cart"]
    And the user checks if "XL" size is available
      # expected_visual_signals: ["XL"]
    When the user selects size "XL"
      # expected_visual_signals: ["XL", "Selected"]
    And the user clicks the "Add to Cart" button
      # expected_visual_signals: ["Added", "Success", "Cart"]

    # Step 4: Verify Cart
    Then the product should be added to the cart
    When the user navigates to the cart page
      # expected_visual_signals: ["My Cart", "Checkout", "Solid Blue Shirt", "XL"]
    Then the cart should display the product with size "XL" and quantity 1
      # expected_visual_signals: ["Solid Blue Shirt", "XL", "1"]
