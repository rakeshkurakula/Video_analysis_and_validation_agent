@nike @search @shoes @size_check
Feature: Search for black running shoes and check size 10 availability on Nike.in
  As a customer of Nike India
  I want to search for black running shoes
  So that I can check if size 10 is available and add it to my cart

  Background:
    Given the user navigates to "https://www.nike.com/in/"
      # expected_visual_signals: ["Nike", "Search", "Bag Items"]
    And the user waits for the homepage to fully load
      # expected_visual_signals: ["Men", "Women", "Kids", "Sale", "SNKRS"]
    And the user closes any promotional popup or banner if present
      # expected_visual_signals: ["close", "×", "X"]

  @shoes @size_check @running
  Scenario: Search for black running shoes, verify size 10 availability, and add to cart
    # Step 1: Search for product
    When the user clicks the search icon or focuses the search input
      # expected_visual_signals: ["Search", "magnifier", "gn-search-input"]
    And the user types "black running shoes" into the search field
      # expected_visual_signals: ["black running shoes"]
    And the user submits the search query
      # expected_visual_signals: ["Results", "products", "Found"]

    # Step 2: Select product
    Then search results page should show matching products
      # expected_visual_signals: ["Running Shoes", "₹", "Add to Cart", "MRP"]
    When the user clicks on the first matching black running shoe product
      # expected_visual_signals: ["Product Details", "Description", "Size", "Available"]

    # Step 3: Check size 10 availability and Add to Cart
    Then the product detail page should display size options
      # expected_visual_signals: ["Size", "7", "8", "9", "10", "11", "Add to Bag"]
    And the user checks if "size 10" is available
      # expected_visual_signals: ["10", "Available"]
    When the user selects size "10"
      # expected_visual_signals: ["10", "Selected"]
    And the user clicks the "Add to Bag" button
      # expected_visual_signals: ["Added", "Success", "View Bag"]

    # Step 4: Verify Cart
    Then the product should be added to the bag
    When the user navigates to the cart page
      # expected_visual_signals: ["Bag Items", "Checkout", "Black Running Shoes", "Size 10"]
    Then the cart should display the product with size "10" and quantity 1
      # expected_visual_signals: ["Black Running Shoes", "Size 10", "1", "₹"]
