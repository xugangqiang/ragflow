import { render, screen } from '@testing-library/react';
import React from 'react';

jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

import {
  LATEST_VERSION_VALUE,
  PipelineVersionSelect,
  filterVersionsByAlias,
  formValueToSelectValue,
  selectValueToFormValue,
} from './index';

describe('filterVersionsByAlias', () => {
  const versions = [
    { id: 'v1', title: 'First', alias: 'prod' },
    { id: 'v2', title: 'Second', alias: null },
    { id: 'v3', title: 'Third', alias: 'PROD-2' },
  ];

  it('returns all versions when the filter is empty or whitespace', () => {
    expect(filterVersionsByAlias(versions, '')).toHaveLength(3);
    expect(filterVersionsByAlias(versions, '   ')).toHaveLength(3);
  });

  it('filters case-insensitively by alias', () => {
    expect(filterVersionsByAlias(versions, 'prod').map((v) => v.id)).toEqual([
      'v1',
      'v3',
    ]);
  });

  it('treats a null alias as non-matching', () => {
    expect(filterVersionsByAlias(versions, 'second')).toHaveLength(0);
  });
});

describe('PipelineVersionSelect', () => {
  it('renders the alias filter input', () => {
    render(
      React.createElement(PipelineVersionSelect, {
        versions: [],
        value: '',
        onChange: jest.fn(),
      }),
    );
    expect(
      screen.getByPlaceholderText('flow.version.filterByAlias'),
    ).toBeInTheDocument();
  });

  it('shows the latest placeholder when no version is pinned', () => {
    render(
      React.createElement(PipelineVersionSelect, {
        versions: [{ id: 'v1', title: 'First', alias: null }],
        value: '',
        onChange: jest.fn(),
      }),
    );
    expect(screen.getByText('flow.version.latest')).toBeInTheDocument();
  });
});

describe('version select value boundary', () => {
  it('maps an empty form value to the sentinel and back', () => {
    expect(formValueToSelectValue('')).toBe(LATEST_VERSION_VALUE);
    expect(selectValueToFormValue(LATEST_VERSION_VALUE)).toBe('');
  });

  it('passes a pinned id through unchanged', () => {
    expect(formValueToSelectValue('v1')).toBe('v1');
    expect(selectValueToFormValue('v1')).toBe('v1');
  });

  it('uses a non-empty sentinel so Radix never sees an empty item value', () => {
    expect(LATEST_VERSION_VALUE.trim()).not.toBe('');
  });
});
