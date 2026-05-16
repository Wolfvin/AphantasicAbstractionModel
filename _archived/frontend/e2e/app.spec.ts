import { test, expect } from '@playwright/test';

test.describe('RSVS Frontend', () => {
  test('has correct title', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/RSVS/i);
  });

  test('renders the graph area', async ({ page }) => {
    await page.goto('/');
    // The 3D canvas should be present
    const canvas = page.locator('canvas').first();
    await expect(canvas).toBeVisible({ timeout: 10000 });
  });

  test('shows input rail', async ({ page }) => {
    await page.goto('/');
    // Input area should be visible
    const input = page.locator('textarea, input[type="text"]').first();
    await expect(input).toBeVisible({ timeout: 10000 });
  });

  test('health check shows backend status', async ({ page }) => {
    // Navigate and check if the page loads without errors
    const response = await page.goto('/');
    expect(response?.ok()).toBeTruthy();
  });
});
